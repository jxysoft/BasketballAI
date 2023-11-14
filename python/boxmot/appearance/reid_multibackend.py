# Mikel Broström 🔥 Yolo Tracking 🧾 AGPL-3.0 license

from collections import OrderedDict, namedtuple
from os.path import exists as file_exists
from pathlib import Path

import cv2
import gdown
import numpy as np
import torch
import torch.nn as nn

from boxmot.appearance.backbones import build_model, get_nr_classes
from boxmot.appearance.reid_model_factory import (get_model_name,
                                                  get_model_url,
                                                  load_pretrained_weights,
                                                  show_downloadable_models)
from boxmot.utils import logger as LOGGER
from boxmot.utils.checks import TestRequirements

tr = TestRequirements()


def check_suffix(file="osnet_x0_25_msmt17.pt", suffix=(".pt",), msg=""):
    # Check file(s) for acceptable suffix
    if file and suffix:
        if isinstance(suffix, str):
            suffix = [suffix]
        for f in file if isinstance(file, (list, tuple)) else [file]:
            s = Path(f).suffix.lower()  # file suffix
            if len(s):
                try:
                    assert s in suffix
                except AssertionError as err:
                    LOGGER.error(f"{err}{f} acceptable suffix is {suffix}")


class ReIDDetectMultiBackend(nn.Module):
    # ReID models MultiBackend class for python inference on various backends
    def __init__(
        self, weights="osnet_x0_25_msmt17.pt", device=torch.device("cpu"), fp16=False
    ):
        super().__init__()

        w = weights[0] if isinstance(weights, list) else weights
        (
            self.pt,
            self.jit,
            self.onnx,
            self.xml,
            self.engine,
            self.tflite,
        ) = self.model_type(w)  # get backend
        self.fp16 = fp16
        self.fp16 &= self.pt or self.jit or self.engine  # FP16
        self.device = device
        self.nhwc = self.tflite  # activate bhwc --> bcwh

        model_name = get_model_name(w)

        if w.suffix == ".pt":
            model_url = get_model_url(w)
            if not file_exists(w) and model_url is not None:
                gdown.download(model_url, str(w), quiet=False)
            elif file_exists(w):
                pass
            else:
                LOGGER.error(
                    f"No URL associated to the chosen StrongSORT weights ({w}). Choose between:"
                )
                show_downloadable_models()
                exit()

        # Build model
        self.model = build_model(
            model_name,
            num_classes=get_nr_classes(w),
            pretrained=not (w and w.is_file()),
            use_gpu=device,
        )

        if self.pt:  # PyTorch
            # populate model arch with weights
            if w and w.is_file() and w.suffix == ".pt":
                load_pretrained_weights(self.model, w)
            self.model.to(device).eval()
            self.model.half() if self.fp16 else self.model.float()
        elif self.onnx:  # ONNX Runtime
            LOGGER.info(f"Loading {w} for ONNX Runtime inference...")
            cuda = torch.cuda.is_available() and device.type != "cpu"
            tr.check_packages(("onnx", "onnxruntime-gpu" if cuda else "onnxruntime", ))
            import onnxruntime

            providers = (
                ["CUDAExecutionProvider", "CPUExecutionProvider"]
                if cuda
                else ["CPUExecutionProvider"]
            )
            self.session = onnxruntime.InferenceSession(str(w), providers=providers)
        else:
            LOGGER.error("This model framework is not supported yet!")
            exit()

    @staticmethod
    def model_type(p="path/to/model.pt"):
        # Return model type from model path, i.e. path='path/to/model.onnx' -> type=onnx
        from . import export_formats

        sf = list(export_formats().Suffix)  # export suffixes
        check_suffix(p, sf)  # checks
        types = [s in Path(p).name for s in sf]
        return types

    def preprocess(self, xyxys, img):
        crops = []
        # dets are of different sizes so batch preprocessing is not possible
        for box in xyxys:
            x1, y1, x2, y2 = box.astype('int')
            # if x1 == x2:
            #     x2 += 1
            crop = img[y1:y2, x1:x2]
            # resize
            crop = cv2.resize(
                crop,
                (128, 256),  # from (x, y) to (128, 256) | (w, h)
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

    def forward(self, im_batch):

        # batch to half
        if self.fp16 and im_batch.dtype != torch.float16:
            im_batch = im_batch.half()

        # torch BCHW to numpy BHWC
        if self.nhwc:
            im_batch = im_batch.permute(0, 2, 3, 1)

        # batch processing
        features = []
        if self.pt:
            features = self.model(im_batch)
        elif self.jit:  # TorchScript
            features = self.model(im_batch)
        elif self.onnx:  # ONNX Runtime
            im_batch = im_batch.cpu().numpy()  # torch to numpy
            features = self.session.run(
                [self.session.get_outputs()[0].name],
                {self.session.get_inputs()[0].name: im_batch},
            )[0]
        elif self.tflite:
            im_batch = im_batch.cpu().numpy()
            inputs = {
                'images': im_batch,
            }
            tf_lite_output = self.tf_lite_model(**inputs)
            features = tf_lite_output['output']

        elif self.engine:  # TensorRT
            if True and im_batch.shape != self.bindings["images"].shape:
                i_in, i_out = (
                    self.model_.get_binding_index(x) for x in ("images", "output")
                )
                self.context.set_binding_shape(
                    i_in, im_batch.shape
                )  # reshape if dynamic
                self.bindings["images"] = self.bindings["images"]._replace(
                    shape=im_batch.shape
                )
                self.bindings["output"].data.resize_(
                    tuple(self.context.get_binding_shape(i_out))
                )
            s = self.bindings["images"].shape
            assert (
                im_batch.shape == s
            ), f"input size {im_batch.shape} {'>' if self.dynamic else 'not equal to'} max model size {s}"
            self.binding_addrs["images"] = int(im_batch.data_ptr())
            self.context.execute_v2(list(self.binding_addrs.values()))
            features = self.bindings["output"].data
        elif self.xml:  # OpenVINO
            im_batch = im_batch.cpu().numpy()  # FP32
            features = self.executable_network([im_batch])[self.output_layer]
        else:
            LOGGER.error(
                "Framework not supported at the moment, leave an enhancement suggestion"
            )
            exit()

        if isinstance(features, (list, tuple)):
            return (
                self.to_numpy(features[0]) if len(features) == 1 else [self.to_numpy(x) for x in features]
            )
        else:
            return self.to_numpy(features)

    def to_numpy(self, x):
        return x.cpu().numpy() if isinstance(x, torch.Tensor) else x

    def warmup(self, imgsz=[(256, 128, 3)]):
        # warmup model by running inference once
        if self.device.type != "cpu":
            im = np.random.randint(0, 255, *imgsz, dtype=np.uint8)
            im = self.preprocess(xyxys=np.array([[0, 0, 128, 256]]), img=im)
            self.forward(im)  # warmup
        else:
            print("no warm for cpu")

    @torch.no_grad()
    def get_features(self, xyxys, img):
        if xyxys.size != 0:
            crops = self.preprocess(xyxys, img)
            features = self.forward(crops)
        else:
            features = np.array([])
        features = features / np.linalg.norm(features)
        return features
