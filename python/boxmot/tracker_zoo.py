# Mikel Broström 🔥 Yolo Tracking 🧾 AGPL-3.0 license

from types import SimpleNamespace

import yaml

from boxmot.utils import BOXMOT


def get_tracker_config(tracker_type):
    tracking_config = \
        BOXMOT /\
        'configs' /\
        (tracker_type + '.yaml')
    return tracking_config


def create_tracker(tracker_type, tracker_config, reid_weights, device, half):

    with open(tracker_config, "r") as f:
        cfg = yaml.load(f.read(), Loader=yaml.FullLoader)
    cfg = SimpleNamespace(**cfg)  # easier dict acces by dot, instead of ['']

    if tracker_type == 'bytetrack':
        from boxmot.trackers.bytetrack.byte_tracker import BYTETracker
        bytetracker = BYTETracker(
            track_thresh=cfg.track_thresh,
            match_thresh=cfg.match_thresh,
            track_buffer=cfg.track_buffer,
            frame_rate=cfg.frame_rate
        )
        return bytetracker

    elif tracker_type == 'botsort':
        from boxmot.trackers.botsort.bot_sort import BoTSORT
        botsort = BoTSORT(
            reid_weights,
            device,
            half,
            track_high_thresh=cfg.track_high_thresh,
            track_low_thresh=cfg.track_low_thresh,
            new_track_thresh=cfg.new_track_thresh,
            track_buffer=cfg.track_buffer,
            match_thresh=cfg.match_thresh,
            proximity_thresh=cfg.proximity_thresh,
            appearance_thresh=cfg.appearance_thresh,
            cmc_method=cfg.cmc_method,
            frame_rate=cfg.frame_rate,
        )
        return botsort
    else:
        print('No such tracker')
        exit()
