from abc import ABC, abstractmethod
from dataclasses import dataclass
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
