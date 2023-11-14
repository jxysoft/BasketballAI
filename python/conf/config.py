from dataclasses import dataclass
from enum import Enum
from typing import Any

# track class enum
class TrackCls(Enum):
    ball = 0
    made = 1
    person = 2
    rim = 3
    shoot = 4

IMG_SIZE=(128, 256) # person and body img size (width, hieght)

# config
tracker_type = 'botsort'
tracking_config = '/Users/jxy/Projects/person/BasketballAI/python/boxmot/configs/botsort.yaml'


@dataclass
class BallEvent(object):
    global_person_id = 0
    global_made_id = 0
    global_shoot_id = 0

    last_last_shoot_time = -1
    last_shoot_time = -1
    last_shoot_id = -1

    last_last_made_time = -1
    last_made_time = -1
    last_made_id = -1
    
    last_store = False
    findPlayerByBody = False
    findPlayerByFace = False
    
    @staticmethod
    def getPersonId():
        BallEvent.global_person_id += 1
        return BallEvent.global_person_id - 1
    
    @staticmethod
    def getMadeId():
        BallEvent.global_made_id += 1
        return BallEvent.global_made_id - 1
    
    @staticmethod
    def getShootId():
        BallEvent.global_shoot_id += 1
        return BallEvent.global_shoot_id - 1
    
# shoot 主要由 时间 + player + ball, shoot track id有可能会间隔不同，但是只要在一定时间内，player+ball相同就是同一个shoot
# 可能ball会无，比如球出手后那段时间
@dataclass
class ShootEvent(BallEvent):
    def __init__(self, shoot_id,  shoot_track_id,  startFrame, player, ball, conf):
        self.shoot_id = shoot_id
        self.shoot_track_id = shoot_track_id
        self.startFrame = startFrame
        self.endFrame = startFrame
        self.player = player
        self.ball = ball
        self.conf = conf
        self.made = False
        self.is_activate = False
        
    def __str__(self):
        return f"{str(self.__dict__)}, player={str(self.player)}"   
    
    def update(self, current_shoot):
        if current_shoot.ball is not None:
            self.ball = current_shoot.ball
        if current_shoot.player is not None:
            self.player = current_shoot.player
        if current_shoot.made:
            self.made = True
        self.shoot_track_id = current_shoot.shoot_track_id
        self.conf = current_shoot.conf
        self.endFrame = current_shoot.endFrame
        self.is_activate = True
    
# made 主要由 时间 + ball， 通过ball 找到对应的player，track id有可能会间隔不同，但是只要在一定时间内，ball相同就是同一个made
class MadeEvent(BallEvent):
    def __init__(self, made_id, made_track_id, startFrame, ball, conf, shoot_info = None):
        self.made_id = made_id
        self.made_track_id = made_track_id
        self.startFrame = startFrame
        self.endFrame = startFrame
        self.ball = ball
        self.conf = conf
        self.shoot_info = shoot_info
        self.is_activate = False
        
    def __str__(self):
        return f"{str(self.__dict__)}, shoot_info={str(self.shoot_info)}"      
        
    def update(self, current_made):
        self.made_track_id = current_made.made_track_id
        if current_made.ball is not None:
            self.ball = current_made.ball
        self.conf = current_made.conf
        if current_made.shoot_info is not None:
            self.shoot_info = current_made.shoot_info
        self.endFrame = current_made.endFrame
        self.is_activate = True
    
@dataclass
class Player(object):
    def __init__(self, id, track_id, name, faceFeat = [], bodyFeat = []) -> None:
        self.id = id
        self.track_id = track_id
        self.name = name
        self.faceFeat = faceFeat
        self.bodyFeat = bodyFeat

    def __str__(self):
        return f"{self.id}, {self.track_id}, {self.name}"
    
@dataclass
class BallResult(object):
    #shoot res: key = shoot id, value = {shoot_time, ball_track_id, player_track_id, confidence, result}
    shoot_res = {}
    #made res: key = made id, value = { made_time, ball_track_id, confidence, player_track_id, shoot_id}
    made_res = {}
    # ball res: key = ball track id, value = {shoot_id}
    ball_res = {}
    
    # person face input, key = name/track_id
    person_faces = {}
    # person res, can be init by input body : key = person id/track_id
    person_res = {}

@dataclass
class BallConfig(object):
    # (shoot begin time - storePre（5s）, made end time + storePost（0.5s) 是要保存的made视频
    # store made  video name = [player_track_id]_[made_id]_[made_time]
    eventInterSeconds = 1 # 1s, 比如两次made之间的间隔不能少于1s
    eventDelaySeconds = 0.25 # 0.25s 这个延迟内都属于同一个事件
    
    storePreSeconds = 5 # 5s shoot事件前面多保留的一点时间， 正常从上上次shoot开始即可; made对应的shoot 时间也使用这个参数
    storePostSeconds = 0.5 # 0.5s made事件后面多保留的一点时间
    storeMaxSeconds = 10 # 10s
    storeMinSeconds= 5 # 5 s
    
    def __init__(self, fps: int = 30, args = None):
        self.fps = fps

        self.eventInterFrames =  int(self.eventInterSeconds * self.fps)
        self.eventDelayFrames =  int(self.eventDelaySeconds * self.fps)
        
        self.storePreFrames = int(self.storePreSeconds * self.fps)
        self.storePostFrames = int(self.storePostSeconds * self.fps)
        self.storeMaxFrames = int(self.storeMaxSeconds * self.fps)
        self.storeMinFrames = int(self.storeMinSeconds * self.fps)
    
