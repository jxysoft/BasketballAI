# Mikel Broström 🔥 Yolo Tracking 🧾 AGPL-3.0 license

__version__ = '10.0.43'

from boxmot.postprocessing.gsi import gsi
from boxmot.tracker_zoo import create_tracker, get_tracker_config
from boxmot.trackers.botsort.bot_sort import BoTSORT
from boxmot.trackers.bytetrack.byte_tracker import BYTETracker

TRACKERS = ['bytetrack', 'botsort']

__all__ = ("__version__",
           "BYTETracker", "BoTSORT",
           "create_tracker", "get_tracker_config", "gsi")
