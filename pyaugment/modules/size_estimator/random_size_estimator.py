from pyaugment.modules.size_estimator.base_size_estimator import BaseSizeEstimator, ObjectSize
import random


class RandomSizeEstimator(BaseSizeEstimator):
    """
    A size estimator that generates random object sizes in pixels within specified height and width ranges.

    This class inherits from BaseSizeEstimator and provides an estimate() method to generate
    random object sizes based on the given height and width ranges.

    Args:
        height_range (tuple): A tuple specifying the minimum and maximum height in pixels (inclusive)
            for generating random object sizes.
        width_range (tuple): A tuple specifying the minimum and maximum width in pixels (inclusive)
            for generating random object sizes.

    Methods:
        estimate() -> ObjectSize:
            Generates a random object size within the specified height and width ranges.
    """

    def __init__(self, height_range: tuple, width_range: tuple) -> None:
        self.height_range = height_range
        self.width_range = width_range
    
    def estimate(self) -> ObjectSize:
        """
        Generate a random object size (in pixels) within the specified height and width ranges.

        Returns:
            ObjectSize: A randomly generated object size with height and width within
            the specified ranges.
        """
        height = random.randint(min(self.height_range), max(self.height_range))
        width = random.randint(min(self.width_range), max(self.width_range))
        object_size = ObjectSize(height, width)
        return object_size