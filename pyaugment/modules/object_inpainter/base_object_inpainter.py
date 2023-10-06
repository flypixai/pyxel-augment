from abc import ABC, abstractmethod
from typing import List, Optional

import numpy

from pyaugment.modules.bbox_generator.base_bbox_generator import RBBox
from pyaugment.modules.region_proposer.base_region_proposer import AnnotatedImage


class BaseObjectInpainter(ABC):
    @abstractmethod
    def __init__(self) -> None:
        pass

    @abstractmethod
    def inpaint_object(
        self,
        background_images: List[AnnotatedImage],
        bbox: List[RBBox],
        text_condition: Optional[str] = None,
        image_condition: Optional[numpy.ndarray] = None,
    ) -> List[numpy.ndarray]:
        pass
