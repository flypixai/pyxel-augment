import math
from typing import List, Tuple, Optional

import cv2
import numpy as np
from PIL import Image
from dataclasses import dataclass



@dataclass
class RBBox:
    x_center: float
    y_center: float
    height: float
    width: float
    alpha: Optional[float] = 0.0


def get_vertex_coordinates(
    x_center: float,
    y_center: float,
    width: float,
    height: float,
    angle_degrees: float,
) -> List[Tuple[float]]:
    angle_radians = math.radians(angle_degrees)
    cos_theta = math.cos(angle_radians)
    sin_theta = math.sin(angle_radians)

    half_width = width / 2
    half_height = height / 2

    x1 = int(x_center + half_width * cos_theta - half_height * sin_theta)
    y1 = int(y_center + half_width * sin_theta + half_height * cos_theta)

    x2 = int(x_center + half_width * cos_theta + half_height * sin_theta)
    y2 = int(y_center + half_width * sin_theta - half_height * cos_theta)

    x3 = int(x_center - half_width * cos_theta + half_height * sin_theta)
    y3 = int(y_center - half_width * sin_theta - half_height * cos_theta)

    x4 = int(x_center - half_width * cos_theta - half_height * sin_theta)
    y4 = int(y_center - half_width * sin_theta + half_height * cos_theta)

    return (x1, y1), (x2, y2), (x3, y3), (x4, y4)


def get_padded_outbounding_bbox(
    rotated_bbox: List[Tuple[float]], min_padding: int
) -> Tuple[float]:
    x_coords, y_coords = zip(*rotated_bbox)
    x_min = min(x_coords)
    x_max = max(x_coords)
    y_min = min(y_coords)
    y_max = max(y_coords)

    x_min -= min_padding
    y_min -= min_padding
    x_max += min_padding
    y_max += min_padding

    width = x_max - x_min
    height = y_max - y_min

    diff = width - height
    if diff > 0:
        y_min -= diff // 2
        y_max += diff // 2
    else:
        x_min -= abs(diff) // 2
        x_max += abs(diff) // 2
    return (x_min, y_min, x_max, y_max)


def transform_bbox_coordinates(
    bbox: List[Tuple[float]], new_coordinates_system: Tuple[float]
) -> np.array:
    x_offset, y_offset, x_max, y_max = new_coordinates_system

    size = np.array([x_max - x_offset, y_max - y_offset])
    offset = np.array([x_offset, y_offset])

    bbox_np = np.array(bbox)

    bbox_transformed_normalized = (bbox_np - offset) / size

    return bbox_transformed_normalized


def draw_rotated_bbox(points: np.array, image_size: tuple) -> Image:
    image_bbox = np.zeros(image_size, dtype=np.uint8)
    points_denormalized = (points * np.array([image_size[0], image_size[1]])).astype(
        np.int32
    )
    cv2.drawContours(image_bbox, [points_denormalized], 0, (255, 255, 255), -1)
    image_bbox = Image.fromarray(image_bbox)
    return image_bbox


def convert_rotated_bbox_to_yolo(rbbox: RBBox, image_size: tuple) -> List:
    rect = get_vertex_coordinates(
        rbbox.x_center, rbbox.y_center, rbbox.width, rbbox.height, rbbox.alpha
    )

    x_values, y_values = zip(*rect)
    x_min, x_max = min(x_values), max(x_values)
    y_min, y_max = min(y_values), max(y_values)

    width = x_max - x_min
    height = y_max - y_min

    x_center = x_min + width / 2
    y_center = y_min + height / 2

    x_center = x_center / image_size[0]
    width = width / image_size[0]
    y_center = y_center / image_size[1]
    height = height / image_size[1]

    yolo_bbox = [x_center, y_center, width, height]
    return yolo_bbox
