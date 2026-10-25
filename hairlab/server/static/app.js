/* HairLab labelling bay.

   One job: decide what a two-second clip shows, several hundred times,
   without waiting. Clips are preloaded a few ahead so a verdict is followed
   immediately by the next clip rather than by a spinner. */

const el = (id) => document.getElementById(id);

const ui = {
  bay: el("bay"),
  stage: el("stage"),
  canvas: el("canvas"),
  wash: el("wash"),
  state: el("state"),
  stateLine: el("stateLine"),
  stateHint: el("stateHint"),
  ticks: el("ticks"),
  frameCount: el("frameCount"),
  verdicts: el("verdicts"),
  clipId: el("clipId"),
  undoBtn: el("undoBtn"),
  railFill: el("railFill"),
  rail: document.querySelector(".rail"),
  tallyDone: el("tallyDone"),
  tallyLeft: el("tallyLeft"),
};

const ctx = ui.canvas.getContext("2d");

const state = {
  labels: [],
  queue: [],          // clips awaiting a verdict, current one first
  loaded: new Map(),  // clip_id -> HTMLImageElement[]
  current: null,
  frame: 0,
  playing: true,
  lastTick: 0,
  busy: false,
};

/* ---------------------------------------------------------------- fetch */

async function api(path, options) {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!response.ok) {
    const detail = await response.json().catch(() => ({}));
    throw new Error(detail.detail || `Запрос не прошёл: ${response.status}`);
  }
  return response.json();
}

function preload(clip) {
  if (state.loaded.has(clip.clip_id)) return;
  const images = clip.frames.map((src) => {
    const img = new Image();
    img.src = src;
    return img;
  });
  state.loaded.set(clip.clip_id, images);
}

/* ------------------------------------------------------------ rendering */

function drawFrame() {
  const clip = state.current;
  if (!clip) return;
  const images = state.loaded.get(clip.clip_id);
  if (!images) return;

  const img = images[state.frame];
  if (img && img.complete && img.naturalWidth) {
    if (ui.canvas.width !== img.naturalWidth) {
      ui.canvas.width = img.naturalWidth;
      ui.canvas.height = img.naturalHeight;
    }
    ctx.drawImage(img, 0, 0, ui.canvas.width, ui.canvas.height);
  }

  ui.frameCount.textContent = `${state.frame + 1}/${clip.frame_count}`;
  const ticks = ui.ticks.children;
  for (let i = 0; i < ticks.length; i += 1) {
    ticks[i].classList.toggle("film__tick--on", i <= state.frame);
  }
}

function buildTicks(count) {
  ui.ticks.replaceChildren();
  for (let i = 0; i < count; i += 1) {
    const tick = document.createElement("span");
    tick.className = "film__tick";
    ui.ticks.appendChild(tick);
  }
}

function loop(now) {
  const clip = state.current;
  if (clip && state.playing) {
    const interval = 1000 / (clip.fps || 10);
    if (now - state.lastTick >= interval) {
      state.lastTick = now;
      state.frame = (state.frame + 1) % clip.frame_count;
      drawFrame();
    }
  }
  requestAnimationFrame(loop);
}

/* --------------------------------------------------------------- screen */

function showClip(clip) {
  state.current = clip;
  state.frame = 0;
  state.playing = true;
  ui.stage.classList.remove("stage--empty");
  ui.state.hidden = true;
  ui.clipId.textContent = clip.clip_id;
  buildTicks(clip.frame_count);
  setVerdictsEnabled(true);
  drawFrame();
}

function showMessage(line, hint) {
  state.current = null;
  ui.stage.classList.add("stage--empty");
  ui.state.hidden = false;
  ui.stateLine.textContent = line;
  ui.stateHint.textContent = hint || "";
  ui.clipId.textContent = "—";
  ui.frameCount.textContent = "—";
  ui.ticks.replaceChildren();
  setVerdictsEnabled(false);
}

function setVerdictsEnabled(enabled) {
  for (const button of ui.verdicts.children) button.disabled = !enabled;
}

function renderVerdicts() {
  ui.verdicts.replaceChildren();
  state.labels.forEach((label, index) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "verdict";
    button.style.setProperty("--hue", `var(--${label.id})`);
    button.dataset.label = label.id;
    button.innerHTML = `
      <span class="verdict__key">${index + 1}</span>
      <span class="verdict__title"></span>
      <span class="verdict__hint"></span>`;
    button.querySelector(".verdict__title").textContent = label.title;
    button.querySelector(".verdict__hint").textContent = label.hint;
    button.addEventListener("click", () => decide(label.id));
    ui.verdicts.appendChild(button);
  });
}

function renderStats(stats) {
  ui.tallyDone.textContent = stats.labelled;
  ui.tallyLeft.textContent = stats.remaining;
  const done = stats.total ? (stats.labelled / stats.total) * 100 : 0;
  ui.railFill.style.width = `${done}%`;
  ui.rail.setAttribute("aria-valuenow", Math.round(done));
  ui.undoBtn.disabled = !stats.can_undo;
}

/* -------------------------------------------------------------- actions */

function flash(labelId) {
  ui.stage.style.color = `var(--${labelId})`;
  ui.stage.classList.remove("stage--decided");
  void ui.stage.offsetWidth;  // restart the animation
  ui.stage.classList.add("stage--decided");

  const button = ui.verdicts.querySelector(`[data-label="${labelId}"]`);
  if (button) {
    button.classList.add("verdict--pressed");
    setTimeout(() => button.classList.remove("verdict--pressed"), 160);
  }
}

async function decide(labelId) {
  if (!state.current || state.busy) return;
  const clipId = state.current.clip_id;
  state.busy = true;
  flash(labelId);

  try {
    await api("/api/labels", {
      method: "POST",
      body: JSON.stringify({ clip_id: clipId, label: labelId }),
    });
    state.loaded.delete(clipId);
    state.queue = state.queue.filter((clip) => clip.clip_id !== clipId);
    advance();
    await refill();
  } catch (error) {
    showMessage("Не удалось сохранить метку", error.message);
  } finally {
    state.busy = false;
  }
}

async function undo() {
  if (state.busy) return;
  state.busy = true;
  try {
    await api("/api/labels/undo", { method: "POST" });
    state.queue = [];
    state.loaded.clear();
    await refill(true);
  } catch (error) {
    if (!String(error.message).includes("Отменять нечего")) {
      showMessage("Не удалось отменить", error.message);
    }
  } finally {
    state.busy = false;
  }
}

function advance() {
  if (state.queue.length) {
    showClip(state.queue[0]);
  } else {
    showMessage("Все клипы размечены", "Включите сбор данных в трее HairGuard — новые клипы появятся здесь.");
  }
}

async function refill(force) {
  const data = await api("/api/queue?limit=4");
  state.queue = data.clips;
  state.queue.forEach(preload);

  if (force || !state.current) {
    if (state.queue.length) showClip(state.queue[0]);
    else showMessage("Все клипы размечены", "Включите сбор данных в трее HairGuard — новые клипы появятся здесь.");
  }
  renderStats(await api("/api/stats"));
}

/* ------------------------------------------------------------- keyboard */

document.addEventListener("keydown", (event) => {
  if (event.metaKey || event.ctrlKey || event.altKey) return;

  const index = Number(event.key) - 1;
  if (Number.isInteger(index) && index >= 0 && index < state.labels.length) {
    event.preventDefault();
    decide(state.labels[index].id);
    return;
  }

  switch (event.key) {
    case " ":
      event.preventDefault();
      state.playing = !state.playing;
      break;
    case "ArrowLeft":
      event.preventDefault();
      if (state.current) {
        state.playing = false;
        state.frame = (state.frame - 1 + state.current.frame_count) % state.current.frame_count;
        drawFrame();
      }
      break;
    case "ArrowRight":
      event.preventDefault();
      if (state.current) {
        state.playing = false;
        state.frame = (state.frame + 1) % state.current.frame_count;
        drawFrame();
      }
      break;
    case "Backspace":
    case "u":
    case "U":
      event.preventDefault();
      undo();
      break;
    default:
      break;
  }
});

ui.undoBtn.addEventListener("click", undo);

/* ----------------------------------------------------------------- boot */

(async function start() {
  try {
    const definitions = await api("/api/labels/definitions");
    state.labels = definitions.labels;
    renderVerdicts();
    await refill(true);
    requestAnimationFrame(loop);
  } catch (error) {
    showMessage("Сервер недоступен", error.message);
  }
})();
