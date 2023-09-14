from abc import ABC, abstractmethod
import numpy

class BaseRegionProposer(ABC):
    """
    An abstract base class for region proposer methods.

    Methods:
        propose_region(image, prompt) -> list:
            Abstract method to propose regions of interest (ROIs) in an image.

            Args:
                image: The input image for which regions of interest are proposed.
                prompt: Additional prompt or information that may guide the region proposal.

            Returns:
                list: a list containing segmentation coordinates of the proposed region.

    Example:
        # Create a custom region proposer by subclassing BaseRegionProposer
        class MyRegionProposer(BaseRegionProposer):
            def propose_region(self, image, prompt) -> list:
                # Implement your region proposal logic here
                # Return a list of proposed regions of interest (ROIs).
                pass
    """
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def propose_region(self, image: numpy.ndarray, prompt) -> list:
        """
        Abstract method to propose regions of interest (ROIs) in an image.

        Subclasses should implement this method to provide custom region proposal logic.

        Args:
            image: The input image for which regions of interest are proposed.
            prompt: Additional prompt or information that may guide the region proposal.

        Returns:
            list: A list of proposed regions of interest (ROIs).
        """
        pass
    def save_results(self, json_file: bool = True, image_file: bool = False):
        pass