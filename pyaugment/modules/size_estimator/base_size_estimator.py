from abc import ABC, abstractmethod
from dataclasses import dataclass

@dataclass
class ObjectSize:
    """
    Represents the size of an object in pixels.

    Attributes:
        height (int): The height of the object in pixels.
        width (int): The width of the object in pixels.
    """
    height: int
    width: int

class BaseSizeEstimator(ABC):
    """
    An abstract base class for size estimators that provide object size in pixels.

    Methods:
        estimate() -> ObjectSize:
            This method should be implemented in concrete subclasses to provide the estimated
            size of an object in pixels.

    Example:
        # Create a custom size estimator by subclassing BaseSizeEstimator and implementing the estimate() method.
        class MySizeEstimator(BaseSizeEstimator):
            def estimate(self) -> ObjectSize:
                # Implement your size estimation logic here
                return ObjectSize(height, width)  # Provide height and width in pixels.
    """
    @abstractmethod
    def estimate(self) -> ObjectSize:
        pass