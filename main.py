import argparse
import json
import os
import random
import shutil
from pathlib import Path

from pyaugment.modules.bbox_generator.random_bbox_generator import RandomRBBoxGenerator
from pyaugment.modules.object_inpainter.canny_controlnet_inpainter import (
    CannyControlNetObjectInpainter,
)
from pyaugment.modules.region_proposer.seem_region_proposer import SEEMRegionProposer
from pyaugment.modules.size_estimator.random_size_estimator import RandomSizeEstimator


def main(args):
    with open(args.object_specifications, "r") as obj_file:
        configs = json.load(obj_file)

    objects = configs["Objects"]
    conditions_path = configs["Conditions"]
    images_path = args.yolo_ds_path + "/images"
    images_path_synthetic = args.yolo_ds_path + "_synthetic"

    n_objects_range = args.n_objects_range

    try:
        shutil.copytree(str(Path(images_path).parent), images_path_synthetic)
    except:
        print("synthtetic folder already exists")

    try:
        os.mkdir(images_path_synthetic + "/labels")
    except:
        print("labels exists")

    config_dir = args.config_dir

    proposer = SEEMRegionProposer(config_file_path=config_dir)
    generator = RandomRBBoxGenerator()

    inpainter = CannyControlNetObjectInpainter(
        controlnet_checkpoint=args.controlnet_checkpoint,
        inpainting_model_checkpoint=args.inpainting_model_checkpoint,
    )

    for obj in objects:
        obj_name = obj["Name"]
        obj_condition_path = Path(conditions_path, obj_name)
        possible_conditions = list(obj_condition_path.glob("*"))
        sample_condition = str(random.sample(possible_conditions, 1)[0])

        size_estimator = RandomSizeEstimator(
            height_range=obj["Height_range"], width_range=obj["Width_range"]
        )

        objsize = size_estimator.estimate()

        output_proposer = proposer.propose_region(
            images_path=images_path_synthetic + "/images", prompt=obj["Background"]
        )
        bbox = generator.generate_bbox(
            proposed_regions=output_proposer,
            object_size=objsize,
            num_objects=n_objects_range,
        )

        generator.save_as_yolo(obj_id=obj["id"], bboxes=bbox, images=output_proposer)

        prompt = f"a top view of a {obj_name} as seen by a satellite image"
        negative_prompt = "vivid colors. shiny. reflection. detailed"

        output_inpainter = inpainter.inpaint_object(
            image_condition_path=sample_condition,
            background_images=output_proposer,
            text_condition=prompt,
            negative_prompt=negative_prompt,
            bboxes=bbox,
            controlnet_conditioning_scale=0.7,
            num_inference_steps=20,
        )
        for i, out in enumerate(output_inpainter):
            file_name = output_proposer[i].file_name.name
            file_path = Path(images_path_synthetic, "images", file_name)
            out.save(file_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Data Augmentation using Canny ControlNet Inpainting"
    )
    parser.add_argument(
        "--object_specifications",
        type=str,
        default="object_specifications.json",
        help="Path to the object specifications JSON file",
    )
    parser.add_argument(
        "--yolo_ds_path",
        type=str,
        default="rotterdam_subset_for_cars_white",
        help="Path to the images directory",
    )
    parser.add_argument(
        "--config_dir",
        type=str,
        default="models/SEEM/seem_focall_lang.yaml",
        help="Path to the SEEM configuration file",
    )
    parser.add_argument(
        "--controlnet_checkpoint",
        type=str,
        default="SyrineKh/construction_waste_color_controlnet",
        help="ControlNet checkpoint",
    )
    parser.add_argument(
        "--inpainting_model_checkpoint",
        type=str,
        default="runwayml/stable-diffusion-inpainting",
        help="Inpainting model checkpoint",
    )
    parser.add_argument(
        "--n_objects_range",
        nargs="+",
        type=int,
        default=[2, 7],
        help="Range of the number of objects",
    )
    args = parser.parse_args()

    main(args)
