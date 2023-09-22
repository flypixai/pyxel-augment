import math
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
from diffusers import (
    ControlNetModel,
    StableDiffusionInpaintPipeline,
    UniPCMultistepScheduler,
)
from diffusers.utils import load_image
from PIL import Image
from src.pipeline_stable_diffusion_controlnet_inpaint import *

from pyaugment.modules.bbox_generator.base_bbox_generator import BBox
from pyaugment.modules.object_inpainter.base_object_inpainter import BaseObjectInpainter

MIN_PADDING = 10


class CannyControlNetObjectInpainter(BaseObjectInpainter):
    def __init__(
        self, controlnet_checkpoint: str, inpainting_model_checkpoint: str
    ) -> None:
        controlnet = ControlNetModel.from_pretrained(
            controlnet_checkpoint, torch_dtype=torch.float16
        )
        self.pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
            inpainting_model_checkpoint,
            controlnet=controlnet,
            torch_dtype=torch.float16,
        )
        self.pipe.scheduler = UniPCMultistepScheduler.from_config(
            self.pipe.scheduler.config
        )
        self.pipe.safety_checker = None
        self.pipe.requires_safety_checker = False
        self.pipe.to("cuda")

    def inpaint_object(
        self,
        background_image: np.ndarray,
        image_condition: np.ndarray,
        text_condition: str,
        bbox: BBox,
        num_inference_steps: Optional[int] = 30,
        controlnet_conditioning_scale: Optional[float] = 0.8,
    ) -> np.ndarray:
        (
            mask_image,
            canny_image,
            context_image,
            context_bbox,
        ) = self._get_controlnet_inputs(background_image, image_condition, bbox)

        generated_image = self.pipe(
            text_condition,
            num_inference_steps=num_inference_steps,
            image=context_image,
            control_image=canny_image,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
            mask_image=mask_image,
            num_images_per_prompt=1,
        ).images[0]

        final_image = self._resize_and_paste(
            context_bbox, generated_image, background_image
        )

        return final_image

    def _get_controlnet_inputs(
        self, background_image: np.ndarray, canny_image: np.ndarray, bbox: BBox
    ):
        rotated_bbox = self._get_rotated_bbox(
            bbox.center_x, bbox.center_y, bbox.width, bbox.height, bbox.alpha
        )
        bbox = self._get_bbox(rotated_bbox)

        context_bbox = self._get_context_bbox(bbox)
        relative_bbox = self._get_relative_position(rotated_bbox, context_bbox)
        mask_image = self._draw_rotated_bbox(relative_bbox)
        context_image = self._crop_and_resize(background_image, context_bbox)
        canny_image = self._rotate_and_center(canny_image, bbox.alpha)

        return mask_image, canny_image, context_image, context_bbox

    def _get_rotated_bbox(
        self,
        center_x: float,
        center_y: float,
        width: float,
        height: float,
        angle_degrees: float,
    ) -> List[Tuple[float]]:
        angle_radians = math.radians(angle_degrees)
        cos_theta = math.cos(angle_radians)
        sin_theta = math.sin(angle_radians)

        half_width = width / 2
        half_height = height / 2

        x1 = int(center_x + half_width * cos_theta - half_height * sin_theta)
        y1 = int(center_y + half_width * sin_theta + half_height * cos_theta)

        x2 = int(center_x + half_width * cos_theta + half_height * sin_theta)
        y2 = int(center_y + half_width * sin_theta - half_height * cos_theta)

        x3 = int(center_x - half_width * cos_theta + half_height * sin_theta)
        y3 = int(center_y - half_width * sin_theta - half_height * cos_theta)

        x4 = int(center_x - half_width * cos_theta - half_height * sin_theta)
        y4 = int(center_y - half_width * sin_theta + half_height * cos_theta)

        return (x1, y1), (x2, y2), (x3, y3), (x4, y4)

    def _get_bbox(self, rotated_bbox: List[Tuple[float]]) -> Tuple[float]:
        x_coords, y_coords = zip(*rotated_bbox)
        x_min = min(x_coords)
        x_max = max(x_coords)
        y_min = min(y_coords)
        y_max = max(y_coords)
        return (x_min, y_min, x_max, y_max)

    def _get_relative_position(
        rotated_bbox: List[Tuple[float]], bbox_inpainting: Tuple[float]
    ) -> List[Tuple[float]]:
        x_offset, y_offset, _, _ = bbox_inpainting
        rotated_bbox_transformed = [
            (x[0] - x_offset, x[1] - y_offset) for x in rotated_bbox
        ]
        width = bbox_inpainting[2] - bbox_inpainting[0]
        height = bbox_inpainting[3] - bbox_inpainting[1]
        rotated_bbox_relative = [
            (x[0] * 512 // width, x[1] * 512 // height)
            for x in rotated_bbox_transformed
        ]
        return rotated_bbox_relative

    def _resize_and_paste(self, bbox, generated_image, background_image):
        x_min, y_min, x_max, y_max = bbox
        desired_size = (x_max - x_min, y_max - y_min)
        resized_image = generated_image.resize(desired_size)
        background_image.paste(resized_image, (x_min, y_min, x_max, y_max))
        return background_image

    def _crop_and_resize(self, image, bbox):
        cropped_image = image.crop(bbox)
        resized_image = cropped_image.resize((512, 512))
        return resized_image

    def _resize_and_paste(self, bbox, generated_image, background_image):
        x_min, y_min, x_max, y_max = bbox
        desired_size = (x_max - x_min, y_max - y_min)
        resized_image = generated_image.resize(desired_size)
        background_image.paste(resized_image, (x_min, y_min, x_max, y_max))
        return background_image

    def _draw_rotated_bbox(points) -> Image:
        image_bbox = np.zeros((512, 512, 3), dtype=np.uint8)
        cv2.drawContours(image_bbox, [np.array(points)], 0, (255, 255, 255), -1)
        image_bbox = Image.fromarray(image_bbox)
        return image_bbox

    def _rotate_and_center(image, angle_degrees, width, height, context_bbox):
        factor = 512 // round(context_bbox[2] - context_bbox[0])
        desired_height = height * factor

        original_width, original_height = image.size
        aspect_ratio = original_width / original_height

        desired_width = int(desired_height * aspect_ratio)
        image_resized = image.resize((desired_width, desired_height))

        background_color = (0, 0, 0)
        background = Image.new("RGB", (512, 512), background_color)

        x_offset = (512 - image_resized.width) // 2
        y_offset = (512 - image_resized.height) // 2

        background.paste(image_resized, (x_offset, y_offset))

        image_np = np.array(background)
        angle_radians = np.radians(angle_degrees)
        rotation_matrix = cv2.getRotationMatrix2D(
            (image_np.shape[1] // 2, image_np.shape[0] // 2), -angle_degrees, 1.0
        )

        rotated_image = cv2.warpAffine(
            image_np, rotation_matrix, (image_np.shape[1], image_np.shape[0])
        )
        rotated_image = Image.fromarray(rotated_image)

        return rotated_image

    def _get_bbox_from_label(image_path):
        label_path = image_path.replace("images", "labels")[:-3] + "txt"
        bboxes = []
        with open(label_path, "r") as file:
            for line in file:
                values = [int(float(val)) for val in line.strip().split()]
                bbox = values[1:]
                bbox.append(0)
                bboxes.append(bbox)
        return bboxes
