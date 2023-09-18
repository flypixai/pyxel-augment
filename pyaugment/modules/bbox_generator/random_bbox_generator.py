import random

import cv2
import numpy
from supervision.detection.core import Detections

from pyaugment.modules.bbox_generator.base_bbox_generator import BaseBBoxGenerator, BBox
from pyaugment.modules.size_estimator.base_size_estimator import ObjectSize


class RandomBBoxGenerator(BaseBBoxGenerator):
    def generate_bbox(
        self, proposed_region: Detections, object_size: ObjectSize
    ) -> BBox:
        proposed_region_bbox = proposed_region.xyxy[0]
        proposed_region_segmentation = proposed_region.mask[0]

        x_start = proposed_region_bbox[0] + object_size.width // 2
        y_start = proposed_region_bbox[1] + object_size.height // 2

        x_end = proposed_region_bbox[2] - object_size.width // 2
        y_end = proposed_region_bbox[3] - object_size.height // 2

        bbox_found = False

        while bbox_found == False:
            x_center = random.randint(int(x_start), int(x_end))
            y_center = random.randint(int(y_start), int(y_end))

            bbox_image = numpy.zeros_like(proposed_region_segmentation, dtype=bool)
            bbox_image[
                int(y_center - object_size.height // 2) : int(
                    y_center + object_size.height // 2
                ),
                int(x_center - object_size.width // 2) : int(
                    x_center + object_size.width // 2
                ),
            ] = True

            intersection = proposed_region_segmentation & bbox_image

            if numpy.count_nonzero(intersection) == numpy.count_nonzero(bbox_image):
                bbox_found = True
            else:
                print("Sampled bbox out of segmented area, trying again...")

        bbox = BBox(
            x_center=x_center,
            y_center=y_center,
            height=object_size.height,
            width=object_size.width,
        )
        return bbox
