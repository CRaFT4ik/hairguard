"""hairlab — data collection, labelling and model training for HairGuard.

Kept separate from the ``hairguard`` package: the running application must
not depend on anything here, while this side is free to reuse the app's
detectors for feature extraction.
"""

__version__ = "0.1.0"
