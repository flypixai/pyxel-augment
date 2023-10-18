from abc import ABC, abstractmethod
from pathlib import Path

from typing import List

from pyaugment.modules.size_estimator.base_size_estimator import ObjectSize
from pyaugment.modules.utils.bbox_transforms import convert_rotated_bbox_to_yolo, RBBox
from pyaugment.modules.region_proposer.base_region_proposer import AnnotatedImage



class BaseRBBoxGenerator(ABC):
    @abstractmethod
    def generate_bbox(
        self, proposed_regions: List[AnnotatedImage], object_size: ObjectSize
    ) -> List[RBBox]:
        pass

    def save_as_yolo(self, obj_id, bboxes: List[RBBox], images: List[AnnotatedImage]):
        image_size = images[0].image_array.shape
        for i, bboxes_per_image in enumerate(bboxes):
            image_name = images[i].file_name
            file_name = str(image_name.name)[:-3] + "txt"
            file_path = Path(image_name.parent.parent, "labels", file_name)
            for bbox in bboxes_per_image:
                bbox_transformed = convert_rotated_bbox_to_yolo(bbox, image_size)
                yolo_annotation = f"{obj_id} {' '.join(map(str, bbox_transformed))}\n"
                with open(file_path, "a") as text_file:
                    text_file.write(yolo_annotation)
