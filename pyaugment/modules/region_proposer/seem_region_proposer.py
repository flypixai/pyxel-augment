import os
import sys
from pathlib import Path
from typing import List

## TODO: find a better solution for this
sys.path.append("/home/ubuntu/Segment-Everything-Everywhere-All-At-Once/demo_code")

import numpy as np
import torch
from PIL import Image
from supervision.detection.core import Detections
from tasks import interactive_infer_image
from utils.arguments import load_opt_from_config_files
from utils.constants import COCO_PANOPTIC_CLASSES
from utils.distributed import init_distributed
from xdecoder import build_model
from xdecoder.BaseModel import BaseModel

from pyaugment.modules.region_proposer.base_region_proposer import (
    AnnotatedImage,
    BaseRegionProposer,
)


class SEEMRegionProposer(BaseRegionProposer):
    def __init__(self, config_file_path: str) -> None:
        opt = load_opt_from_config_files(config_file_path)
        opt = init_distributed(opt)
        if "focalt" in config_file_path:
            pretrained_pth = Path("seem_focalt_v2.pt")
        elif "focal" in config_file_path:
            pretrained_pth = Path("seem_focall_v1.pt")
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

    def propose_region(
        self,
        images_path: str,
        prompt: list,
    ) -> List[AnnotatedImage]:
        super().propose_region(images_path=images_path, prompt=prompt)
        prompt = [c + "." for c in prompt]

        image_list = Path(images_path).glob("*")
        annotated_images = []

        for image in image_list:
            detections = self._get_detections(image, prompt)
            annotated_image = AnnotatedImage(file_name=image, detections=detections)
            annotated_images.append(annotated_image)

        return annotated_images

    def _get_detections(self, image_path: str, prompt: List[str]) -> Detections:
        image = Image.open(image_path)
        input = {"image": image, "mask": None}
        masks = np.empty(shape=(len(prompt), image.size[0], image.size[1]))
        classes = np.empty(shape=(len(prompt)))

        for i, cat in enumerate(prompt):
            output = interactive_infer_image(
                model=self.model,
                audio_model=None,
                image=input,
                tasks=["Text"],
                reftxt=cat,
            )
            output = output[0].resize(image.size)
            class_id = self.object_categories[cat[:-1]]
            masks[i, :, :] = np.array(output)[:, :, 0]
            classes[i] = class_id
        xyxy = np.zeros((len(prompt), 4))
        detections = Detections(
            xyxy=xyxy,
            mask=masks.astype(np.uint8),
            class_id=classes,
        )

        return detections
