import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional

import cv2
import numpy
from supervision.detection.core import Detections


@dataclass
class AnnotatedImage:
    file_name: str
    image_array: numpy.ndarray
    detections: Optional[Detections] = None


class BaseRegionProposer(ABC):
    """
    An abstract base class for region proposer methods.

    Methods:
        propose_region(image_path, prompt) -> List[AnnotatedImage]:
            Abstract method to propose regions of interest (ROIs) in an image.

            Args:
                images_path: path of the images for which regions of interest are proposed.
                prompt: Additional prompt or information that may guide the region proposal.

            Returns:
                list: a list containing segmentation coordinates of the proposed region.

    Example:
        # Create a custom region proposer by subclassing BaseRegionProposer
        class MyRegionProposer(BaseRegionProposer):
            def propose_region(self, image, prompt) -> List[AnnotatedImage]:
                # Implement your region proposal logic here
                # Return a list of proposed regions of interest (ROIs).
                pass
    """

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def propose_region(self, images_path: str, prompt) -> List[AnnotatedImage]:
        """
        Abstract method to propose regions of interest (ROIs) in an image.

        Subclasses should implement this method to provide custom region proposal logic.

        Args:
            images_path: path of the images for which regions of interest are proposed.
            prompt: Additional prompt or information that may guide the region proposal.

        Returns:
            list: A list of proposed regions of interest (ROIs).
        """
        self.object_categories = {
            class_name: class_id for class_id, class_name in enumerate(prompt)
        }

    def save_results(
        self, json_file: bool, image_file: bool, detections: List[AnnotatedImage]
    ):
        if json_file:
            data = {
                "categories": self.object_categories,
                "annotations": [
                    {
                        "image": detection.file_name,
                        "segmentation": self._transform_mask_to_polygone(
                            detection.detections.mask
                        ),
                        "classes": [
                            int(class_id) for class_id in detection.detections.class_id
                        ],
                    }
                    for detection in detections
                ],
            }
            with open("gsam_output.json", "w+") as json_file:
                json.dump(data, json_file, indent=4)

        if image_file:
            for img_detect in detections:
                for i, segmentation in enumerate(img_detect.detections.mask):
                    segmentation_image = numpy.uint8(segmentation) * 255
                    cv2.imwrite(
                        img_detect.file_name[:-3] + f"_{i}.jpg", segmentation_image
                    )

    def _transform_mask_to_polygone(self, masks):
        segmentation_all = []
        try:
            for mask in masks:
                contours, hierarchy = cv2.findContours(
                    mask.astype(numpy.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
                )
                segmentation = []

                for contour in contours:
                    contour = contour.flatten().tolist()
                    if len(contour) > 4:
                        segmentation.append(contour)
                if len(segmentation) == 0:
                    continue
                segmentation_all.append(segmentation)
        except:
            return segmentation_all
        return segmentation_all
