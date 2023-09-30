from abc import ABC, abstractmethod
from typing import Optional

import numpy

from pyaugment.modules.bbox_generator.base_bbox_generator import RBBox


class BaseObjectInpainter(ABC):
    @abstractmethod
    def __init__(self) -> None:
        pass

    @abstractmethod
    def inpaint_object(
        self,
        background_image: numpy.ndarray,
        bbox: RBBox,
        text_condition: Optional[str] = None,
        image_condition: Optional[numpy.ndarray] = None,
    ) -> numpy.ndarray:
        pass
