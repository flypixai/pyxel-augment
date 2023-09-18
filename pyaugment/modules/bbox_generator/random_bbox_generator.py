import random

import numpy
from supervision.detection.core import Detections

from pyaugment.modules.bbox_generator.base_bbox_generator import BaseBBoxGenerator, BBox
from pyaugment.modules.size_estimator.base_size_estimator import ObjectSize


class RandomBBoxGenerator(BaseBBoxGenerator):
    def generate_bbox(proposed_region: Detections, object_size: ObjectSize) -> BBox:
        proposed_region_bbox = proposed_region.xyxy[0]
        proposed_region_segmentation = proposed_region.mask[0]

        x_start = proposed_region_bbox[0][0] + object_size.width // 2
        y_start = proposed_region_bbox[0][1] + object_size.height // 2

        x_end = proposed_region_bbox[0][2] - object_size.width // 2
        y_end = proposed_region_bbox[0][3] - object_size.height // 2

        x_range = range(x_start, y_start)
        y_range = range(x_end, y_end)

        bbox_found = False

        while bbox_found == False:
            x_center = random.sample(x_range)
            y_center = random.sample(y_range)

            bbox_image = numpy.zeros(shape=proposed_region_segmentation.shape)
            bbox_image = bbox_image[
                x_center - object_size.width, x_center + object_size.width
            ][y_center - object_size.height, y_center + object_size.height]

            intersection = proposed_region_segmentation & bbox_image

            if numpy.count_nonzero(intersection) == numpy.count_nonzero(bbox_image):
                bbox_found = True

        bbox = BBox(
            x_center=x_center,
            y_center=y_center,
            height=object_size.height,
            width=object_size.width,
        )

        return bbox
