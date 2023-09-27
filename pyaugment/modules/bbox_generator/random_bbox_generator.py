import random
from typing import List

import numpy
from shapely import MultiPolygon, Polygon, buffer
from skimage import measure
from supervision.detection.core import Detections

from pyaugment.modules.bbox_generator.base_bbox_generator import BaseBBoxGenerator, BBox
from pyaugment.modules.size_estimator.base_size_estimator import ObjectSize


class RandomBBoxGenerator(BaseBBoxGenerator):
    def generate_bbox(
        self, proposed_region: Detections, object_size: ObjectSize
    ) -> BBox:
        proposed_region_segmentation = proposed_region.mask[0]

        contours = self._get_segmentation_contour(proposed_region_segmentation)

        buffer_threshold = numpy.hypot(object_size.height, object_size.width) / 2

        inner_contours = self._get_inner_segmentation_contour(
            contours, buffer_threshold
        )

        (
            x_center,
            y_center,
        ) = random.choice(inner_contours[0])

        bbox = BBox(
            x_center=x_center,
            y_center=y_center,
            height=object_size.height,
            width=object_size.width,
            alpha=int(random.uniform(0, 180)),
        )
        return bbox, inner_contours

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
