from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import numpy
from supervision.detection.core import Detections

from pyaugment.modules.size_estimator.base_size_estimator import ObjectSize


@dataclass
class BBox:
    x_center: float
    y_center: float
    height: float
    width: float
    alpha: Optional[float] = 0.0


class BaseBBoxGenerator(ABC):
    @abstractmethod
    def generate_bbox(
        self, proposed_region: Detections, object_size: ObjectSize
    ) -> BBox:
        pass
