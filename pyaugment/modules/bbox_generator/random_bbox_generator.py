import math
import random
from pathlib import Path
from typing import List

import cv2
import numpy
from shapely import MultiPolygon, Polygon, buffer
from skimage import measure
from supervision.detection.core import Detections

from pyaugment.modules.bbox_generator.base_bbox_generator import (
    BaseRBBoxGenerator,
    RBBox,
)
from pyaugment.modules.region_proposer.base_region_proposer import AnnotatedImage
from pyaugment.modules.size_estimator.base_size_estimator import ObjectSize

BUFFER = 3


class RandomRBBoxGenerator(BaseRBBoxGenerator):
    def generate_bbox(
        self,
        proposed_regions: List[AnnotatedImage],
        object_size: ObjectSize,
        num_objects: List[int],
    ) -> RBBox:
        bboxes = []
        for proposed_region in proposed_regions:
            bboxes_per_image = []
            proposed_region_segmentation = proposed_region.detections.mask[0]

            num_object_rand = random.randint(num_objects[0], num_objects[1])

            for i in range(num_object_rand):
                contours = self._get_segmentation_contour(proposed_region_segmentation)

                buffer_threshold = (
                    numpy.hypot(object_size.height, object_size.width) * BUFFER / 2
                )

                inner_contours = self._get_inner_segmentation_contour(
                    contours, buffer_threshold
                )
                # TODO find a better way to do this without taking only the first contour
                try:
                    (
                        x_center,
                        y_center,
                    ) = random.choice(random.choice(inner_contours))
                except:
                    print(f"{proposed_region.file_name} skipped")
                    break

                bbox = RBBox(
                    x_center=x_center,
                    y_center=y_center,
                    height=object_size.height,
                    width=object_size.width,
                    alpha=int(random.uniform(0, 180)),
                )
                bboxes_per_image.append(bbox)
                proposed_region_segmentation = self._update_region(
                    proposed_region_segmentation, bbox, inner_contours[0]
                )
                new_image = numpy.copy(proposed_region_segmentation)
                for cont in inner_contours:
                    vertices = numpy.array(cont, dtype=numpy.int32)
                    vertices = vertices.reshape((-1, 1, 2))
                    new_image = cv2.polylines(
                        new_image, [vertices], isClosed=True, color=(0, 255, 0)
                    )
            bboxes.append(bboxes_per_image)
        return bboxes

    def _update_region(self, region, bbox, exterior):
        center = (bbox.x_center, bbox.y_center)
        size = (bbox.width, bbox.height)
        angle = bbox.alpha

        radius = int(math.hypot(bbox.width, bbox.height)) * BUFFER

        rect = cv2.boxPoints(((center[0], center[1]), (size[0], size[1]), angle))
        rect = numpy.int0(rect)

        center = numpy.int0(center)

        new_region = cv2.circle(region, center, radius, color=(0, 0, 0), thickness=-1)
        return new_region

    def _get_segmentation_contour(
        self,
        segmentation: numpy.ndarray,
    ) -> List[Polygon]:
        segmentation_bordered = numpy.zeros(segmentation.shape, dtype=numpy.uint8)
        segmentation_bordered[1:-1, 1:-1] = segmentation[1:-1, 1:-1]
        contours = measure.find_contours(segmentation_bordered.astype(int), 0.5)
        contours_as_polygons = [Polygon(contour) for contour in contours]
        return contours_as_polygons

    def _get_inner_segmentation_contour(
        self, contours: List[Polygon], buffer_threshold: float
    ) -> List[numpy.ndarray]:
        new_contours = []
        for contour in contours:
            new_contour = buffer(
                contour, -buffer_threshold, cap_style="flat", join_style="bevel"
            )
            # TODO: find a way to avoid hardcoding 0.5
            new_contour = new_contour.simplify(0.5)

            if isinstance(new_contour, Polygon):
                new_contour = MultiPolygon([new_contour])

            for polygon_part in new_contour.geoms:
                new_contour_coordinates = numpy.array(polygon_part.exterior.coords)
                if not new_contour.is_empty:
                    new_contour_coordinates = new_contour_coordinates[
                        :,
                        [
                            1,
                            0,
                        ],  ## TODO: look deeper into this, why/when should we switch coordinates
                    ]
                    new_contours.append(new_contour_coordinates)
        return new_contours
