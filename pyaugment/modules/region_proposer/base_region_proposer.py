from abc import ABC, abstractmethod
import numpy
from supervision.detection.core import Detections
from typing import List, Optional
from dataclasses import dataclass
import json
from pycocotools import mask as coco_mask
import cv2


@dataclass
class ImageDetection:
    file_name: str
    image_array: numpy.ndarray
    detections: Optional[Detections] = None

@dataclass
class ImageDetectionList:
    detected_object: str
    detections_list: List[ImageDetection]

class BaseRegionProposer(ABC):
    """
    An abstract base class for region proposer methods.

    Methods:
        propose_region(image_path, prompt) -> ImageDetectionList:
            Abstract method to propose regions of interest (ROIs) in an image.

            Args:
                images_path: path of the images for which regions of interest are proposed.
                prompt: Additional prompt or information that may guide the region proposal.

            Returns:
                list: a list containing segmentation coordinates of the proposed region.

    Example:
        # Create a custom region proposer by subclassing BaseRegionProposer
        class MyRegionProposer(BaseRegionProposer):
            def propose_region(self, image, prompt) -> ImageDetectionList:
                # Implement your region proposal logic here
                # Return a list of proposed regions of interest (ROIs).
                pass
    """
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def propose_region(self, images_path: str, prompt) -> ImageDetectionList:
        """
        Abstract method to propose regions of interest (ROIs) in an image.

        Subclasses should implement this method to provide custom region proposal logic.

        Args:
            images_path: path of the images for which regions of interest are proposed.
            prompt: Additional prompt or information that may guide the region proposal.

        Returns:
            list: A list of proposed regions of interest (ROIs).
        """
        pass
    def save_results(self, 
                     json_file: bool, 
                     image_file: bool,
                     detection_list: ImageDetectionList):
        if json_file: 
            data = {
                "object": detection_list.detected_object,
                "annotations": [
                    {
                        "image": detection.file_name,
                        "segmentation": self._transform_mask_to_polygone(detection.detections.mask)
                    }
                    for detection in detection_list.detections_list
                ]
            }
            with open("gsamoutput.json", "w+") as json_file:
                json.dump(data, json_file, indent=4)
    
    def _transform_mask_to_polygone(self, masks):
        segmentation_all = []
        for mask in masks:
            contours, hierarchy = cv2.findContours(mask.astype(numpy.uint8), cv2.RETR_TREE,
                                                                cv2.CHAIN_APPROX_SIMPLE)
            segmentation = []

            for contour in contours:
                contour = contour.flatten().tolist()
                if len(contour) > 4:
                    segmentation.append(contour)
            if len(segmentation) == 0:
                continue
            segmentation_all.append(segmentation)
        return segmentation_all