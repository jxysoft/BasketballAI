import argparse
from pathlib import Path
import time
import queue

import torch
from ultralytics import YOLO

import supervision as sv
from supervision import VideoInfo

from tqdm import tqdm

from boxmot.tracker_zoo import create_tracker
from boxmot.trackers.botsort.bot_sort import STrack
from boxmot.utils.matching import (embedding_distance, linear_assignment)

from conf.config import BallConfig, BallEvent, BallResult, MadeEvent, Player, ShootEvent
from utils.image_utils import *
from utils.utils import *

def getDevice(args):
    if args.device == 'cpu':
        device = torch.device('cpu')
    elif args.device == 'mps':
        device = torch.device('mps')
    else:
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
    return device

def getDetectModel(args):
    model = YOLO(args.detect_model)
    import logging
    # disable yolo log
    logger = logging.getLogger('ultralytics')
    logger.disabled = True
    return model


def getReidModel(detect_model, args):
    tracker = create_tracker(
                str(args.tracking_method),
                args.tracking_config,
                args.reid_model,
                getDevice(args),
                args.half
            )
    return tracker.model

def findPersonByTrackId(track_id):
    for _, person in BallResult.person_res.items():
        if person.track_id == track_id:
            return person
    return None

def handleShoot(ballConfig, current_frame_id, result, shoot_boxes, reid_model, frame):
        # how to deal with multiple shoot boxes?
        if BallEvent.last_shoot_time <= 0 or (current_frame_id - BallEvent.last_shoot_time) > ballConfig.eventInterFrames:
            # new shoot found
            BallEvent.global_shoot_id +=1
            BallEvent.last_last_shoot_time = BallEvent.last_shoot_time
        BallEvent.last_shoot_time = current_frame_id
        shoot_id = BallEvent.global_shoot_id
        BallEvent.last_shoot_id = shoot_id
        shoot_time = current_frame_id
        shoot_box = shoot_boxes[0]
        # find the person made the shoot which will be used in made
        player_founded, player_box = find_shooter(result, shoot_box)
        player = None
        # todo find player by body or not
        if player_founded:
            # first find player by track_id if exsist in player_res
            track_id = int(player_box.id.item())
            player = None # findPersonByTrackId(track_id), tracker id may change
            if player is None:
                # then find the player by body and face, or add to player res if is empyt
                # todo: face
                body_emmedings = getBodyFeatures()
                    
                dets = shoot_box.data
                features = reid_model.get_features(dets[:, 0:4].numpy(), np.asarray(frame))
                detections = [STrack(det.numpy(), feat) for det, feat in zip(dets, features) ]
                emb_dists = embedding_distance(body_emmedings, detections) / 2.0
                matches, _, _ = linear_assignment(
                    emb_dists, thresh=0.2
                ) 
                if len(matches) > 0: # find the player
                    # print(f"matches={matches}, emb_dists={emb_dists}")
                    player = BallResult.person_res[matches[0][0]]
                    player.track_id = int(player_box.id.item())
                elif BallEvent.findPlayerByBody == False:
                    #print("auto add player id=" + str(track_id))
                    # auto add to player res
                    id = BallEvent.getPersonId()
                    player = Player(id = id, track_id= track_id, name = str(id), bodyFeat=detections)
                    BallResult.person_res[id] = player
        # find the ball made the shoot which will be used in made
        ball_founded, ball_box = find_ball(result, shoot_box)
        ball_track_id = int(ball_box.id.item()) if ball_founded else None
        current_shoot = ShootEvent(shoot_id, int(shoot_box.id.item()), current_frame_id, 
                                   player=player, ball=ball_track_id, conf = shoot_box.conf)
        
        # save the shoot info
        last_shoot = BallResult.shoot_res.get(shoot_id)
        if last_shoot is None:
            BallResult.shoot_res[shoot_id] = current_shoot
        else:
            last_shoot.update(current_shoot)

        # debug print
        print("find shoot time=" + str(shoot_time) + " last shoot_time=" + str(BallEvent.last_last_shoot_time) + " id=" + str(shoot_id) + " current=" + str(current_shoot.__dict__) + ", pre=" + str(last_shoot))        

def getShootByBall(ballConfig, current_frame_id):
    # find the shoot for the made by ball id
        # if ball_track_id is not None:
        #     shoot_id = ball_res[ball_track_id]['shoot_id'] if ball_res.get(ball_track_id) is not None else None
        #     shoot_inf = shoot_res[shoot_id] if shoot_id is not None else None
        #     player_track_id = shoot_inf['player_id'] if shoot_inf is not None else None
        #     shoot_time = shoot_inf['shoot_time'] if shoot_inf is not None else None
        
        # 没找到则直接使用上一次的
        # if shoot_inf is None and (current_time - last_shoot_time) < delayFrame:
        #     shoot_id = last_shoot_id
        #     shoot_inf = shoot_res[shoot_id]
        #     player_track_id = shoot_inf['player_id']    
        #     ball_track_id = shoot_inf['ball_id'] if ball_track_id is None else ball_track_id
        #     shoot_time = shoot_inf['shoot_time']
    # current only use last shoot
    shoot_id = BallEvent.last_shoot_id if (current_frame_id - BallEvent.last_shoot_time) < ballConfig.storePreFrames else  None
    if shoot_id is not None:
        return BallResult.shoot_res[shoot_id]
    else:
        return None

def handleMade(ballConfig, current_frame_id, result, made_boxes):
        # how to deal with multiple made boxes
        if BallEvent.last_made_time <= 0 or (current_frame_id - BallEvent.last_made_time) > ballConfig.eventInterFrames:
            # new made found
            BallEvent.last_store = False
            BallEvent.global_made_id +=1
        BallEvent.last_made_time = current_frame_id
        made_id = BallEvent.global_made_id
        BallEvent.last_made_id = made_id
        made_time = current_frame_id
        made_box = made_boxes[0]
        made_track_id = int(made_box.id.item())
        made_conf = made_box.conf
        # find the ball in the made
        ball_founded, ball_box = find_ball(result, made_box)
        ball_track_id = int(ball_box.id.item()) if ball_founded else None
        
        shoot_info = getShootByBall(ballConfig, current_frame_id)
        current_made = MadeEvent(made_id, made_track_id, current_frame_id, ball=ball_track_id, conf = made_conf, shoot_info=shoot_info)
        last_made = BallResult.made_res.get(made_id)
        if last_made is None:
            BallResult.made_res[made_id] = current_made
        else:
            last_made.update(current_made)
            
        # update shoot result
        if shoot_info is not None:
            shoot_info.made = True
            
        # debug print
        print("find made time=" + str(made_time) + " id=" + str(made_id) + " currentMade=" + str(current_made) + ", preMade=" + str(last_made) + ", shoot=" + str(shoot_info))
        
def getVideos(args):
    # get input videos
    input_videos = str(args.input)
    videos = sv.list_files_with_extensions(
        directory= input_videos,
        extensions=["mp4", "MP4", "avi", "mov"])
    videos.sort(reverse=True)
    print(f"input={input_videos}, videos len={len(videos)}, videos={videos}")
    return videos

def getFaceAndBody(detect_model, reid_model, args):
    # get input faces
    if args.filterByFace == True:
        input_face = str(args.input_person) + "/face"
        face_files = sv.list_files_with_extensions( 
            directory= input_face,
            extensions=["jpg", "png"])
        
        for image in face_files:
            name = image.stem.split("_")[0]
            img = cv2.imread(str(image))
            person = Player(id = -1, track_id = -1, name = name, faceFeat=[img])
            old_person = BallResult.person_faces.get(name)
            if old_person is None:
                BallResult.person_faces[name] = person
            else:
                old_person.faceFeat.append(img)
        if (len(BallResult.person_faces) > 0):
            BallEvent.findPlayerByFace = True
    
    # get input bodys
    if args.filterByBody == True:
        input_body = str(args.input_person) + "/body"
        body_files = sv.list_files_with_extensions(
            directory= input_body,
            extensions=["jpg", "png"])  
        body_images = [ cv2.imread(str(img)) for img in body_files]
        results = detect_model.track(source=body_images, conf = args.conf, classes=[TrackCls.person.value]) 
        
        for file_path, result, img in zip(body_files, results, body_images):
            dets = result.boxes.data
            features = reid_model.get_features(dets[:, 0:4].numpy(), np.asarray(img))
            name = file_path.stem.split("_")[0]
            strack = STrack(dets[0].numpy(), features[0])
            id = BallEvent.global_person_id
            BallEvent.global_person_id = BallEvent.global_person_id + 1
            person = Player(id = id, track_id= -1, name = name, bodyFeat=[strack])
            BallResult.person_res[id] = person
            
        if (len(BallResult.person_res) >= 0):
            BallEvent.findPlayerByBody = True

def getFaceFeatures():
    # todo
    pass

def getBodyFeatures():
    rst_body = []
    for _, person in BallResult.person_res.items():
        rst_body.append(person.bodyFeat[0])
    return rst_body

def handleStoreVideos(target_path_prefix, current_frame_id, ballConfig, frame_cache, source_video_info):
    # handle store video
    if BallEvent.last_made_time > 0 and current_frame_id - BallEvent.last_made_time > ballConfig.storePostFrames and BallEvent.last_store == False:
        last_made_info = BallResult.made_res[BallEvent.last_made_id]
        
        stime = last_made_info.shoot_info.endFrame if last_made_info.shoot_info is not None else 0
        needStoreSize = min(current_frame_id - stime + ballConfig.storePreFrames, frame_cache.qsize())
        needStoreSize = min(max(current_frame_id - BallEvent.last_last_shoot_time, ballConfig.storeMinFrames), needStoreSize)
        for idx in range(frame_cache.qsize() - needStoreSize):
            frame = frame_cache.get()
        
        player = last_made_info.shoot_info.player if last_made_info.shoot_info is not None else None
        if player is None: # filter this made
            BallEvent.last_store = True 
            return
        pid = last_made_info.shoot_info.player.name
        # trigger store: last is made and current is not made
        # store made  video name = [player_track_id]_[made_id]_[made_time]
        target_path = f"{target_path_prefix}/{pid}_{str(BallEvent.last_made_id)}_{BallEvent.last_made_time}.mp4"
        with sv.VideoSink(target_path=target_path, video_info=source_video_info) as sink:
            for idx in range(frame_cache.qsize()):
                frame = frame_cache.get()
                sink.write_frame(frame=frame)
        print(f"finished store video {target_path},needStoreSize={needStoreSize},cur={current_frame_id}, last={stime}, llst={BallEvent.last_last_shoot_time}")
        BallEvent.last_store = True                
def isValidFrame(args, current_frame_id):
    if current_frame_id < args.start_frame_id:
        return False 
    # handle frame range
    if len(args.frame_ranges) > 1:
        for idx in range(0, len(args.frame_ranges), 2):
            if current_frame_id >= args.frame_ranges[idx] and current_frame_id <= args.frame_ranges[idx + 1]:
                return True
        return False
    else:
        return True

@torch.no_grad()
def run(args):
    model = getDetectModel(args)
    reid_model = getReidModel(model, args)
    # get face and body
    getFaceAndBody(model,reid_model, args)
    
    videos = getVideos(args)
    if (len(videos) == 0):
        print("no videos found")
        return
    
    # assume all videos has same fps and image size
    source_path = str(videos[0]) # only one video for test
    source_video_info = VideoInfo.from_video_path(video_path=source_path)
    ballConfig = BallConfig(source_video_info.fps, args)
    frame_cache = queue.Queue(ballConfig.storeMaxFrames)
    target_path_prefix = str(args.output)
    
    current_frame_id = 0
    s_total_time = time.time()
    # begin read videos
    for video_path in videos:
        source_video_info = VideoInfo.from_video_path(video_path=str(video_path))
        s_time = time.time()
        totalIt = (source_video_info.total_frames - 1)// args.vid_stride + 1
        for frame in tqdm(sv.get_video_frames_generator(source_path=str(video_path), stride=args.vid_stride), total=totalIt):
            if(frame_cache.full()):
                frame_cache.get()
            frame_cache.put(frame)
            
            if isValidFrame(args, current_frame_id) == False:
                current_frame_id += args.vid_stride
                continue 

            # conf= args.conf, iou= args.iou,
            result = model.track(source=frame, imgsz=(source_video_info.height, source_video_info.width), classes=args.classes, persist = True)[0]
            
            shoot_founded, shoot_boxes = shoot_found(result)
            made_founded, made_boxes = made_found(result)
            # print(f"frame={current_frame_id}, shoot_founded={shoot_founded}, made_founded={made_founded}")
            # handle shoot info
            if shoot_founded:
                handleShoot(ballConfig, current_frame_id, result, shoot_boxes, reid_model, frame)
                
            # handle made info
            if made_founded:
                handleMade(ballConfig, current_frame_id, result, made_boxes)
            else: 
                handleStoreVideos(target_path_prefix, current_frame_id, ballConfig, frame_cache, source_video_info)
            current_frame_id +=  args.vid_stride
        e_time = time.time()
        print(f"time={e_time-s_time}, video={str(video_path)},find result=" + str(BallResult.made_res))
    e_total_time = time.time()
    print(f"time={e_total_time-s_total_time}, find result len={len(BallResult.made_res)}") 
def parse_opt():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--detect_model', type=Path, default='./models/yolov8x_best.pt',help='defect model path')
    parser.add_argument('--reid_model', type=Path, default='./models/clip_market1501.pt', help='reid model path')
    parser.add_argument('--tracking_method', type=str, default='botsort', help='botsort, bytetrack')
    parser.add_argument('--tracking_config', type=Path, default='./python/boxmot/configs/botsort.yaml', help='tracking config path')
    parser.add_argument('--input_person', type=Path, default = "./input/", help='input person path')
    parser.add_argument('--input', type=Path, default = "./input/test/videos", help='input path')
    parser.add_argument('--output', type=Path, default = "./output/highlights", help='output path')  
    parser.add_argument('--vid_stride', type=int, default=1, help='video frame-rate stride')
    parser.add_argument('--start_frame_id', type=int, default=-1, help='video start frame id') 
    # 560,750,3000,3220,3900,4200
    parser.add_argument('--frame_ranges', nargs='+', type=int, default=[], help='video include frame range, must be pair')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu or mps')
    
    parser.add_argument('--conf', type=float, default=0.5, help='confidence threshold')
    parser.add_argument('--iou', type=float, default=0.7, help='intersection over union (IoU) threshold for NMS')
    parser.add_argument('--classes', nargs='+', type=int, default=[0,1,2,3,4], help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--filterByBody', action='store_false', default=True, help='filter made case by person body')
    parser.add_argument('--filterByFace', action='store_false', default=True, help='filter made case by person face')
    opt = parser.parse_args()
    return opt


if __name__ == "__main__":
    opt = parse_opt()
    run(opt)