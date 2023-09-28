from pathlib import Path
from typing import List

import numpy as np
import torch
from PIL import Image
from supervision.detection.core import Detections
from xdecoder import build_model
from xdecoder.BaseModel import BaseModel

from pyaugment.modules.region_proposer.base_region_proposer import (
    AnnotatedImage,
    BaseRegionProposer,
)


class GroundedSAMRegionProposer(BaseRegionProposer):
    def __init__(self, config_file_path: str) -> None:
        opt = load_opt_from_config_files(conf_files)
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
            image_path = Path(images_path, image)
            detections = self._get_detections(image_path, prompt)
            annotated_image = AnnotatedImage(
                images_path=image_path, detections=detections
            )
            annotated_images.append(annotated_image)

        return annotated_images

    def _get_detections(self, image_path: str, prompt: List[str]) -> Detections:
        input = {"image": Image.open(image_path), "mask": None}
        masks = np.empty_like([])
        classes = np.empty_like([])

        for cat in prompt:
            output = interactive_infer_image(
                model=self.model,
                audio_model=None,
                image=input,
                tasks=["Text"],
                reftxt=cat,
            )
            class_id = self.object_categories[prompt[:-1]]
            masks = np.append(masks, output)
            classes = np.append(classes, class_id)

        xyxy = np.zeros((len(prompt), 4))
        detections = Detections(
            xyxy=xyxy,
            mask=masks,
            class_id=classes,
        )

        return detections
