import argparse
from pathlib import Path
import torch
from ultralytics import YOLO
from PIL import Image
import supervision as sv
from utils.image_utils import get_size, extract_body

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')


@torch.no_grad()
def run(args):
    yolo = YOLO(args.detect_model)
    
    # handle images
    input_images = str(args.input)
    images = sv.list_files_with_extensions(
        directory= input_images,
        extensions=["png", "jpg", "jpg"])
    print(f"input={input_images}, images len={len(images)}")
    
    for img_path in images:
        img = Image.open(img_path)
        # widht, height
        image_size = get_size(img) 
        # only track person
        results = yolo.track(source = [img], imgsz = (image_size[1], image_size[0]), classes = [2])
        boxes = results[0].boxes
        for i, box in enumerate(boxes):
            print(f'body及概率:i={i} , box={box}')
            save_path= str(args.output) + f'/body_{img_path.stem}_{i}.png'
            body = extract_body(img, box.xyxy[0].numpy(), save_path=save_path)
            
    # handle videos
    # input_videos = str(args.input) + "/videos"
    # videos = sv.list_files_with_extensions(
    #     directory= input_videos,
    #     extensions=["mp4", "avi", "mov"])
    # print(f"input={input_videos}, videos len={len(videos)}")
    
def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--detect_model', type=Path, default='./models/yolov8x_best.pt', help='defect model path')
    parser.add_argument('--input', type=Path, default = "./input/images", help='input image path')
    parser.add_argument('--output', type=Path, default = "./output/body", help='output image path')  

    opt = parser.parse_args()
    return opt


if __name__ == "__main__":
    opt = parse_opt()
    run(opt)