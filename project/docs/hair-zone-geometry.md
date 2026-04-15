# Hair Zone Geometry — Mathematical Model

## Problem

Determine the 2D region on a camera image where the user's hair exists, given arbitrary head orientation and camera position. The region must adapt to head tilt (roll), forward lean (pitch), and sideways turn (yaw).

## Input Landmarks

From MediaPipe PoseLandmarker (normalized 0-1 screen coordinates, Y increases downward):

- **N** — nose
- **L** — left ear
- **R** — right ear
- **SL** — left shoulder
- **SR** — right shoulder

Each landmark has a confidence score. Additionally, FaceLandmarker provides a forehead contour for secondary filtering.

## Algorithm

### Step 1: Camera baseline from shoulders

The shoulder line defines the camera's notion of "horizontal" — works regardless of camera position (front, side, above).

```
camera_horizontal = normalize(SR - SL)
```

**Guard:** If either shoulder confidence < 0.5, fall back to screen horizontal `(1, 0)`.

### Step 2: Head orientation vectors

```
ear_vec = R - L
ear_distance = |ear_vec|
```

**Guard:** If either ear confidence < 0.5, or `ear_distance < shoulder_distance * 0.2`, fall back to `camera_horizontal` as the head horizontal.

```
head_horizontal = normalize(ear_vec)
```

### Step 3: Head center and scale

```
head_center = midpoint(L, R) = (L + R) / 2
shoulder_distance = |SR - SL|
head_scale = max(ear_distance, shoulder_distance * 0.45)
```

The coefficient 0.45 is from anthropometric data: head width ≈ 45% of biacromial breadth. The `max` ensures that when the head turns sideways (ears foreshorten), the shoulder-based estimate takes over.

### Step 4: Up vector (toward crown)

In 2D screen coordinates (Y increases downward), rotating a vector 90° CCW gives the visual "upward" perpendicular:

```
up_candidate = (-ear_vec.y, ear_vec.x)   # 90° CCW in screen coords = visual "up"
up_candidate = normalize(up_candidate)
```

**Sign determination using shoulders:** The head center is above the shoulder center in screen space. We use this to verify the sign:

```
shoulder_center = midpoint(SL, SR)
down_vec = shoulder_center - head_center
```

If `dot(up_candidate, down_vec) > 0`, then `up_candidate` points toward shoulders (down) — flip it:

```
if dot(up_candidate, down_vec) > 0:
    up_candidate = -up_candidate
```

**Fallback for top-down camera:** If `|down_vec| < 0.05 * head_scale`, the head and shoulders overlap in projection (camera above). Fall back to screen-up:

```
if |down_vec| < 0.05 * head_scale:
    up_candidate = (0, -1)  # screen-up (negative Y)
```

Final: `up_vec = normalize(up_candidate)`

### Step 5: Pitch detection (decomposed)

Raw `face_vec = N - head_center` conflates yaw and pitch. Decompose:

```
face_vec = N - head_center

# Project face_vec onto up_vec to get signed pitch component.
# Positive = nose is below ear midpoint along up_vec (neutral/leaning back)
# Negative = nose is above ear midpoint (extreme back-lean — looking at ceiling)
# Near zero = nose at ear level (leaning forward — face down)
pitch_component = -dot(face_vec, up_vec)   # negated so forward-lean gives small/negative values
pitch_ratio = pitch_component / head_scale
```

Clamp: `pitch_ratio = clamp(pitch_ratio, -0.3, 0.6)`

- Neutral pose: pitch_ratio ≈ 0.3 (nose below ears)
- Leaning forward: pitch_ratio → 0 or negative (nose rises toward ear level)
- Leaning back: pitch_ratio → 0.5+ (nose drops further below ears)

```
# Pitch factor: expand hair zone when leaning forward (back of head exposed)
pitch_factor = 1.0 + max(0, 0.3 - pitch_ratio) * 1.5
```

### Step 6: Hair zone as rotated ellipse

The hair zone is an ellipse centered above the head center, aligned with the head's orientation:

```
# Offset from head_center to hair_center (in up_vec direction)
base_offset = head_scale * 0.5
pitch_offset_bonus = max(0, 0.3 - pitch_ratio) * head_scale * 0.3
hair_center = head_center + up_vec * (base_offset + pitch_offset_bonus)

# Ellipse semi-axes
semi_width = head_scale * 0.55    # along head_horizontal (ear-to-ear direction)
semi_height = head_scale * 0.6 * pitch_factor   # along up_vec (toward crown)
```

Note: semi_height > semi_width because hair extends more vertically (forehead to nape) than horizontally.

### Step 7: Point-in-ellipse test

To check if point P = (x, y) is inside the hair zone:

```
d = P - hair_center

# Project onto ellipse axes
proj_w = dot(d, head_horizontal) / semi_width
proj_h = dot(d, up_vec) / semi_height

in_zone = (proj_w² + proj_h²) <= 1.0
```

### Step 8: Forehead filter (optional refinement)

If FaceLandmarker is available, use the forehead contour to exclude the face area from the ellipse. This prevents false triggering when touching the forehead (not hair):

```
if face_detected AND face.is_above_forehead(x, y) is False:
    return False  # point is on face, not hair
```

This filter is additive — it only removes false positives, never blocks true ones.

## Step 9: Temporal smoothing

Landmark jitter causes the ellipse to vibrate frame-to-frame. Apply exponential moving average (EMA) to all computed values:

```
alpha = 0.3  # smoothing constant (new = alpha * current + (1 - alpha) * previous)

hair_center = ema(hair_center, alpha)
semi_width = ema(semi_width, alpha)
semi_height = ema(semi_height, alpha)
up_vec = normalize(ema(up_vec, alpha))  # interpolate then re-normalize
head_horizontal = normalize(ema(head_horizontal, alpha))
```

At ~15 FPS, alpha=0.3 gives ~3-frame effective window — responsive enough for real-time detection while eliminating single-frame jitter.

On the first frame (no previous values), use the current frame's values directly (no smoothing).

## Visual Representation

The hair zone on the debug view:
- Draw the ellipse outline (rotated to match head orientation)
- Fill with semi-transparent color
- Mark hair_center with a cross
- Draw up_vec as an arrow from head_center

## Edge Cases and Fallbacks

| Case | Symptom | Mitigation |
|---|---|---|
| Camera from above | `\|down_vec\|` near zero | Fall back to screen-up `(0, -1)` |
| One ear occluded | `ear_distance` very small | Use `shoulder_distance * 0.45` as head_scale; use `camera_horizontal` for head axis |
| Head turned 90° | Ears foreshorten, `ear_vec` shrinks | `max(ear_distance, shoulder_distance * 0.45)` covers this |
| Head fully forward | Nose overlaps ear midpoint | `pitch_factor` increases, expanding zone toward back of head |
| Leaning back | Nose above ear midpoint | `pitch_ratio` increases, `pitch_factor` stays 1.0 (no expansion needed) |
| Both ears occluded | `ear_vec` and `head_center` unreliable | Set `head_center = N` (nose), `head_scale = shoulder_distance * 0.45`, `head_horizontal = camera_horizontal` |
| One shoulder occluded | `camera_horizontal` unreliable | Fall back to screen horizontal `(1, 0)` |

## Constants Summary

| Constant | Value | Source |
|---|---|---|
| Shoulder-to-head width ratio | 0.45 | Anthropometric data |
| Ear distance minimum | `shoulder_distance * 0.2` | Guard against ear foreshortening |
| Hair center offset | `0.5 * head_scale` | Crown is ~50% above ear midpoint |
| Ellipse semi-width | `0.55 * head_scale` | Hair width ≈ head width |
| Ellipse semi-height | `0.6 * head_scale * pitch_factor` | Hair extends more vertically |
| Pitch expansion threshold | 0.3 | Below this pitch_ratio, head is leaning forward |
| down_vec minimum | `0.05 * head_scale` | Guard for top-down camera |
| Confidence threshold | 0.5 | For landmark reliability |
