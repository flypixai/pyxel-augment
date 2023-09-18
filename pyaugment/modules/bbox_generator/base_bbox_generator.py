from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import numpy

from pyaugment.modules.size_estimator.base_size_estimator import ObjectSize


@dataclass
class BBox:
    x_center: float
    y_center: float
    height: float
    width: float
    alpha: Optional[float] = 0


class BaseBBoxGenerator(ABC):
    @abstractmethod
    def generate_bbox(
        proposed_region: numpy.ndarray, object_size: ObjectSize
    ) -> BBox:  ## TODO: extend return more than one bbox at a time?
        pass
