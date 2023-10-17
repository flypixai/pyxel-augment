from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy
from supervision.detection.core import Detections

from pyaugment.modules.region_proposer.base_region_proposer import AnnotatedImage
from pyaugment.modules.size_estimator.base_size_estimator import ObjectSize


@dataclass
class RBBox:
    x_center: float
    y_center: float
    height: float
    width: float
    alpha: Optional[float] = 0.0


class BaseRBBoxGenerator(ABC):
    @abstractmethod
    def generate_bbox(
        self, proposed_regions: List[AnnotatedImage], object_size: ObjectSize
    ) -> List[RBBox]:
        pass

    def save_as_yolo(self, obj_id, bboxes: List[RBBox], images: List[AnnotatedImage]):
        for i, bboxes_per_image in enumerate(bboxes):
            image_name = images[i].file_name
            file_name = str(image_name.name)[:-3] + "txt"
            file_path = Path(image_name.parent.parent, "labels", file_name)
            for bbox in bboxes_per_image:
                yolo_annotation = f"{obj_id} {bbox.x_center} {bbox.y_center} {bbox.width} {bbox.height} {bbox.alpha} \n"
                with open(file_path, "a") as text_file:
                    text_file.write(yolo_annotation)
