from pathlib import Path
from typing import List

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from skimage.transform import resize
from supervision.detection.core import Detections
from torchvision import transforms
from tqdm import tqdm

from pyaugment.modules.region_proposer.base_region_proposer import (
    AnnotatedImage,
    BaseRegionProposer,
)
from pyaugment.modules.utils.SEEM.utils.arguments import load_opt_from_config_files
from pyaugment.modules.utils.SEEM.utils.constants import COCO_PANOPTIC_CLASSES
from pyaugment.modules.utils.SEEM.utils.distributed import init_distributed
from pyaugment.modules.utils.SEEM.xdecoder import build_model
from pyaugment.modules.utils.SEEM.xdecoder.BaseModel import BaseModel
from pyaugment.modules.utils.SEEM.xdecoder.language.loss import vl_similarity


class SEEMRegionProposer(BaseRegionProposer):
    def __init__(self, config_file_path: str) -> None:
        if not torch.cuda.is_available():
            print(
                "ERROR: GPU resources are not available "
                "Please to ensure that a compatible GPU is installed "
                "and properly configured to use SEEMRegionproposer"
            )
            return None
        opt = load_opt_from_config_files(config_file_path)
        opt = init_distributed(opt)
        if "focalt" in config_file_path:
            pretrained_pth = Path(config_file_path).parent / "seem_focalt_v1.pt"
        elif "focal" in config_file_path:
            pretrained_pth = Path(config_file_path).parent / "seem_focall_v0.pt"
        self.model = (
            BaseModel(opt, build_model(opt))
            .from_pretrained(pretrained_pth)
            .eval()
            .cuda()
        )
        with torch.no_grad():
            self.model.model.sem_seg_head.predictor.lang_encoder.get_text_embeddings(
                COCO_PANOPTIC_CLASSES + ["background"], is_eval=True
            )
        self.model.model.task_switch["spatial"] = False
        self.model.model.task_switch["visual"] = False
        self.model.model.task_switch["grounding"] = True
        self.model.model.task_switch["audio"] = False

        self.input_transform = transforms.Compose(
            [transforms.Resize(512, interpolation=Image.BICUBIC)]
        )

    def propose_region(
        self,
        images_path: str,
        prompt: list,
    ) -> List[AnnotatedImage]:
        super().propose_region(images_path=images_path, prompt=prompt)
        prompt = [c + "." for c in prompt]
        images_path = Path(images_path)
        image_list = images_path.glob("*")
        annotated_images = []
        print("Searching for background regions...")
        for image in tqdm(image_list):
            detections = self._get_detections(image, prompt)
            if detections is None:
                continue
            image_array = np.asarray(Image.open(image))
            annotated_image = AnnotatedImage(
                file_name=image, detections=detections, image_array=image_array
            )
            annotated_images.append(annotated_image)

        return annotated_images

    def _get_detections(self, image_path: str, prompt: List[str]) -> Detections:
        image = Image.open(image_path)
        masks = np.empty(shape=(len(prompt), image.size[0], image.size[1]))
        classes = np.empty(shape=(len(prompt)))

        for i, cat in enumerate(prompt):
            output = self._infere_mask(
                image=image,
                reftxt=cat,
            )
            output = resize(output, image.size, anti_aliasing=True)
            class_id = self.object_categories[cat[:-1]]
            masks[i, :, :] = np.array(output)[:, :]
            classes[i] = class_id
        xyxy = np.zeros((len(prompt), 4))

        mask_is_empty = (masks == np.zeros(shape=masks.shape)).all()

        if mask_is_empty:
            return None

        detections = Detections(
            xyxy=xyxy,
            mask=masks.astype(np.uint8),
            class_id=classes,
        )

        return detections

    def _infere_mask(self, image: Image, reftxt: str) -> np.ndarray:
        image_ori = self.input_transform(image)
        width = image_ori.size[0]
        height = image_ori.size[1]

        image_ori = np.asarray(image_ori)
        image = torch.from_numpy(image_ori.copy()).permute(2, 0, 1).cuda()

        data = {
            "image": image,
            "height": height,
            "width": width,
            "text": [reftxt],
        }

        results, image_size, extra = self.model.model.evaluate_demo([data])

        pred_masks = results["pred_masks"][0]
        v_emb = results["pred_captions"][0]
        t_emb = extra["grounding_class"]

        t_emb = t_emb / (t_emb.norm(dim=-1, keepdim=True) + 1e-7)
        v_emb = v_emb / (v_emb.norm(dim=-1, keepdim=True) + 1e-7)

        temperature = self.model.model.sem_seg_head.predictor.lang_encoder.logit_scale
        out_prob = vl_similarity(v_emb, t_emb, temperature=temperature)
        matched_id = out_prob.max(0)[1]
        pred_masks_pos = pred_masks[matched_id, :, :]
        pred_masks_pos = (
            (
                F.interpolate(pred_masks_pos[None,], image_size[-2:], mode="bilinear")[
                    0, :, : data["height"], : data["width"]
                ]
                > 0.0
            )
            .float()
            .cpu()
            .numpy()
        )
        torch.cuda.empty_cache()
        image = np.squeeze(pred_masks_pos * 255, axis=0)
        return image
