import math
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
from diffusers import ControlNetModel, UniPCMultistepScheduler
from diffusers.utils import load_image
from PIL import Image

from pyaugment.modules.bbox_generator.base_bbox_generator import RBBox
from pyaugment.modules.object_inpainter.base_object_inpainter import BaseObjectInpainter
from pyaugment.modules.utils.bbox_transforms import (
    draw_rotated_bbox,
    get_padded_outbounding_bbox,
    get_vertex_coordinates,
    transform_bbox_coordinates,
)
from pyaugment.modules.utils.pipeline_stable_diffusion_controlnet_inpaint import (
    StableDiffusionControlNetInpaintPipeline,
)

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
        background_image_path: str,
        image_condition_path: str,
        text_condition: str,
        bbox: RBBox,
        num_inference_steps: Optional[int] = 30,
        controlnet_conditioning_scale: Optional[float] = 0.8,
    ) -> np.ndarray:
        background_image = load_image(background_image_path)
        image_condition = load_image(image_condition_path)

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
        self, background_image: Image, canny_image: Image, bbox: RBBox
    ):
        bbox_vertices = get_vertex_coordinates(
            bbox.x_center, bbox.y_center, bbox.width, bbox.height, bbox.alpha
        )
        context_bbox = get_padded_outbounding_bbox(bbox_vertices, MIN_PADDING)
        relative_bbox = transform_bbox_coordinates(
            bbox=bbox_vertices, new_coordinates_system=context_bbox
        )
        mask_image = draw_rotated_bbox(relative_bbox, background_image.size)
        context_image = self._crop_and_resize(background_image, context_bbox)
        canny_image = self._rotate_and_center(
            canny_image,
            targe_image_side_length=context_image.size[0],
            angle_degrees=bbox.alpha,
            height=bbox.height,
            context_bbox=context_bbox,
        )

        return mask_image, canny_image, context_image, context_bbox

    def _resize_and_paste(
        self, bbox: Tuple, generated_image: Image, background_image: Image
    ) -> Image:
        x_min, y_min, x_max, y_max = bbox
        desired_size = (x_max - x_min, y_max - y_min)
        resized_image = generated_image.resize(desired_size)
        background_image.paste(resized_image, (x_min, y_min, x_max, y_max))
        return background_image

    def _crop_and_resize(self, image: Image, bbox):
        cropped_image = image.crop(bbox)
        resized_image = cropped_image.resize((512, 512))
        return resized_image

    def _rotate_and_center(
        self,
        image: Image,
        targe_image_side_length: int,
        angle_degrees: float,
        height: int,
        context_bbox: Tuple[float],
    ):
        factor = targe_image_side_length // round(context_bbox[2] - context_bbox[0])
        desired_height = height * factor
        original_width, original_height = image.size
        aspect_ratio = original_width / original_height

        desired_width = int(desired_height * aspect_ratio)
        image_resized = image.resize((desired_width, desired_height))

        background_color = (0, 0, 0)
        background = Image.new(
            "RGB", (targe_image_side_length, targe_image_side_length), background_color
        )

        x_offset = (targe_image_side_length - image_resized.width) // 2
        y_offset = (targe_image_side_length - image_resized.height) // 2

        background.paste(image_resized, (x_offset, y_offset))

        image_np = np.array(background)
        rotation_matrix = cv2.getRotationMatrix2D(
            (image_np.shape[1] // 2, image_np.shape[0] // 2), -angle_degrees, 1.0
        )

        rotated_image = cv2.warpAffine(
            image_np, rotation_matrix, (image_np.shape[1], image_np.shape[0])
        )
        rotated_image = Image.fromarray(rotated_image)

        return rotated_image
