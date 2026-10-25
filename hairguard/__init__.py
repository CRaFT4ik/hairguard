"""HairGuard — webcam-based hair-pulling detector.

Importing this package quiets the native logging of MediaPipe, TensorFlow
Lite and absl before any of them are loaded. They announce themselves with
a dozen INFO and WARNING lines on every start — XNNPACK delegates, feedback
tensors, ROI projection — none of which the person running the app can act
on, and which drown the messages that do matter.

The environment variables only take effect before the libraries load, which
is why this lives in the package __init__ rather than in the detectors.

Set HAIRGUARD_VERBOSE=1 to keep the native logs when debugging.
"""
import os

__version__ = "0.1.0"


def _quiet_native_logging() -> None:
    if os.environ.get("HAIRGUARD_VERBOSE"):
        return
    # glog (MediaPipe C++): 0=INFO 1=WARNING 2=ERROR 3=FATAL
    os.environ.setdefault("GLOG_minloglevel", "2")
    os.environ.setdefault("GLOG_stderrthreshold", "2")
    # TensorFlow Lite / XNNPACK delegate chatter
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    # absl, which MediaPipe's Python layer routes through
    os.environ.setdefault("ABSL_MIN_LOG_LEVEL", "2")
    try:
        from absl import logging as absl_logging

        absl_logging.set_verbosity(absl_logging.ERROR)
    except Exception:
        pass


_quiet_native_logging()
