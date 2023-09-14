from abc import ABC, abstractmethod
import numpy
from supervision.detection.core import Detections
from typing import List, Optional
from dataclasses import dataclass


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
    def save_results(self, json_file: bool = True, image_file: bool = False):
        pass