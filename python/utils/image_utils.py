import torch
from torch.nn.functional import interpolate
from torchvision.transforms import functional as F
from PIL import Image
import numpy as np
import os

# OpenCV is optional, but required if using numpy arrays instead of PIL
try:
    import cv2
except:
    pass

def save_img(img, path):
    if isinstance(img, np.ndarray):
        cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    else:
        img.save(path)
        
def imresample(img, sz):
    im_data = interpolate(img, size=sz, mode="area")
    return im_data

def get_size(img):
    if isinstance(img, (np.ndarray, torch.Tensor)):
        return img.shape[1::-1]
    else:
        return img.size


def crop_resize(img, box, image_size):
    if isinstance(img, np.ndarray):
        img = img[box[1]:box[3], box[0]:box[2]]
        out = cv2.resize(
            img,
            (image_size[0], image_size[1]),
            interpolation=cv2.INTER_AREA
        ).copy()
    elif isinstance(img, torch.Tensor):
        img = img[box[1]:box[3], box[0]:box[2]]
        out = imresample(
            img.permute(2, 0, 1).unsqueeze(0).float(),
            (image_size[0], image_size[1])
        ).byte().squeeze(0).permute(1, 2, 0)
    else:
        out = img.crop(box).copy().resize(image_size, Image.BILINEAR)
    return out

def extract_box(img, box, image_size=160, margin=0, save_path=None):
    """Extract face + margin from PIL Image given bounding box.
    
    Arguments:
        img {PIL.Image} -- A PIL Image.
        box {numpy.ndarray} -- Four-element bounding box.
        image_size {int} -- Output image size in pixels. The image will be square.
        margin {int} -- Margin to add to bounding box, in terms of pixels in the final image. 
            Note that the application of the margin differs slightly from the davidsandberg/facenet
            repo, which applies the margin to the original image before resizing, making the margin
            dependent on the original image size.
        save_path {str} -- Save path for extracted face image. (default: {None})
    
    Returns:
        torch.tensor -- tensor representing the extracted face.
    """
    margin = [
        margin * (box[2] - box[0]) / (image_size - margin),
        margin * (box[3] - box[1]) / (image_size - margin),
    ]
    raw_image_size = get_size(img)
    box = [
        int(max(box[0] - margin[0] / 2, 0)),
        int(max(box[1] - margin[1] / 2, 0)),
        int(min(box[2] + margin[0] / 2, raw_image_size[0])),
        int(min(box[3] + margin[1] / 2, raw_image_size[1])),
    ]

    face = crop_resize(img, box, image_size)

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path) + "/", exist_ok=True)
        save_img(face, save_path)

    face = F.to_tensor(np.float32(face))

    return face

def extract_face(img, box, image_size=160, margin=0, save_path=None):
    """Extract face + margin from PIL Image given bounding box.
    
    Arguments:
        img {PIL.Image} -- A PIL Image.
        box {numpy.ndarray} -- Four-element bounding box.
        image_size {int} -- Output image size in pixels. The image will be square.
        margin {int} -- Margin to add to bounding box, in terms of pixels in the final image. 
            Note that the application of the margin differs slightly from the davidsandberg/facenet
            repo, which applies the margin to the original image before resizing, making the margin
            dependent on the original image size.
        save_path {str} -- Save path for extracted face image. (default: {None})
    
    Returns:
        torch.tensor -- tensor representing the extracted face.
    """
    margin = [
        margin * (box[2] - box[0]) / (image_size - margin),
        margin * (box[3] - box[1]) / (image_size - margin),
    ]
    raw_image_size = get_size(img)
    box = [
        int(max(box[0] - margin[0] / 2, 0)),
        int(max(box[1] - margin[1] / 2, 0)),
        int(min(box[2] + margin[0] / 2, raw_image_size[0])),
        int(min(box[3] + margin[1] / 2, raw_image_size[1])),
    ]

    face = crop_resize(img, box, image_size)

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path) + "/", exist_ok=True)
        save_img(face, save_path)

    face = F.to_tensor(np.float32(face))

    return face

# ReID models use body to match, 
def extract_body(img, box, image_size=(128, 256), margin=0, save_path=None):
    """Extract face + margin from PIL Image given bounding box.
    
    Arguments:
        img {PIL.Image} -- A PIL Image.
        box {numpy.ndarray} -- Four-element bounding box.
        image_size {int} -- Output image size in pixels. (w, h)
        margin {int} -- Margin to add to bounding box, in terms of pixels in the final image. 
            Note that the application of the margin differs slightly from the davidsandberg/facenet
            repo, which applies the margin to the original image before resizing, making the margin
            dependent on the original image size.
        save_path {str} -- Save path for extracted face image. (default: {None})
    
    Returns:
        torch.tensor -- tensor representing the extracted face.
    """
    # margin = [
    #     margin * (box[2] - box[0]) / (image_size - margin),
    #     margin * (box[3] - box[1]) / (image_size - margin),
    # ]
    # raw_image_size = get_size(img)
    # box = [
    #     int(max(box[0] - margin[0] / 2, 0)),
    #     int(max(box[1] - margin[1] / 2, 0)),
    #     int(min(box[2] + margin[0] / 2, raw_image_size[0])),
    #     int(min(box[3] + margin[1] / 2, raw_image_size[1])),
    # ]
    face = crop_resize(img, box.astype('int'), image_size)

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path) + "/", exist_ok=True)
        save_img(face, save_path)

    face = F.to_tensor(np.float32(face))

    return face

def preprocess(img, xyxys):
        crops = []
        # dets are of different sizes so batch preprocessing is not possible
        for box in xyxys:
            x1, y1, x2, y2 = box.astype('int')
            crop = img[y1:y2, x1:x2]
            # resize
            crop = cv2.resize(
                crop,
                (224, 224),  # from (x, y) to (128, 256) | (w, h)
                interpolation=cv2.INTER_LINEAR,
            )

            # (cv2) BGR 2 (PIL) RGB. The ReID models have been trained with this channel order
            crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

            # normalization
            crop = crop / 255

            # standardization (RGB channel order)
            crop = crop - np.array([0.485, 0.456, 0.406])
            crop = crop / np.array([0.229, 0.224, 0.225])

            crop = torch.from_numpy(crop).float()
            crops.append(crop)

        crops = torch.stack(crops, dim=0)
        crops = torch.permute(crops, (0, 3, 1, 2))
        crops = crops.to(dtype=torch.half if self.fp16 else torch.float, device=self.device)

        return crops