from pathlib import Path

import cv2
import numpy
import torch
import torchvision

# TODO: Installing the next two is from source, running this necessiates
# cloning grounded_segment_anything, is there a better way of doing this?
from groundingdino.util.inference import Model as GDinoModel
from segment_anything import SamPredictor, sam_model_registry
from supervision.detection.core import Detections

from pyaugment.modules.region_proposer.base_region_proposer import (
    BaseRegionProposer,
    ImageDetection,
    ImageDetectionList,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class GroundedSAMRegionProposer(BaseRegionProposer):
    """
    A region proposer that combines object detection with object segmentation using Grounding Dino and SAM models.

    Parameters:
        gdino_config_path (str): The path to the Grounding Dino model configuration file.
        gdino_ckpt_path (str): The path to the Grounding Dino checkpoint file.
        sam_encoder_version (str): The version of SAM encoder to use.
        sam_ckpt_path (str): The path to SAM checkpoint file.

    Methods:
        propose_region(images_path: str, prompt: list, box_threshold: float = 0.3, text_threshold: float = 0.25) -> ImageDetectionList:
            Detects objects in a set of images using Grounding Dino, performs non-maximum suppression (NMS) on the detections,
            and segments the objects using SAM.

        __detect_objects_all(images_path: str, classes: list, box_threshold: float, text_threshold: float) -> ImageDetectionList:
            Detects objects in a set of images using the Grounding Dino.

        __reduce_bboxes_all(detection_list: ImageDetectionList) -> ImageDetectionList:
            Applies non-maximum suppression (NMS) to reduce overlapping bounding boxes in a list of detections.

        __segment_objects_all(detections_list: ImageDetectionList) -> ImageDetectionList:
            Segments objects in a list of detections using SAM.

        __detect_objects_all(sam_predictor, image: numpy.ndarray, detections: numpy.ndarray) -> Detections:
            Segments objects in a single image using SAM.

        __reduce_bboxes(detections: Detections) -> Detections:
            Applies non-maximum suppression (NMS) to reduce overlapping bounding boxes in a single detection.

    Note:
        This class extends the BaseRegionProposer class and provides methods for proposing regions by
        combining object detection and segmentation using GDino and SAM models.
    """

    def __init__(
        self, gdino_config_path, gdino_ckpt_path, sam_encoder_version, sam_ckpt_path
    ) -> None:
        super().__init__()
        self.gdino_config_path = gdino_config_path
        self.gdino_ckpt_path = gdino_ckpt_path
        self.sam_ckpt_path = sam_ckpt_path
        self.sam_encoder_version = sam_encoder_version

    def __reduce_bboxes(self, detections: Detections) -> Detections:
        NMS_THRESHOLD = 0.8  ## TODO: find a better way to introduce this
        print(f"Before NMS: {len(detections.xyxy)} boxes")
        nms_idx = (
            torchvision.ops.nms(
                torch.from_numpy(detections.xyxy),
                torch.from_numpy(detections.confidence),
                NMS_THRESHOLD,
            )
            .numpy()
            .tolist()
        )

        detections.xyxy = detections.xyxy[nms_idx]
        detections.confidence = detections.confidence[nms_idx]
        detections.class_id = detections.class_id[nms_idx]

        print(f"After NMS: {len(detections.xyxy)} boxes")

        return detections

    def __segment_objects(
        self, sam_predictor, image: numpy.ndarray, detections: numpy.ndarray
    ) -> Detections:
        # segment
        sam_predictor.set_image(image)
        result_masks = []
        for box in detections.xyxy:
            masks, scores, logits = sam_predictor.predict(
                box=box, multimask_output=True
            )
            index = numpy.argmax(scores)
            result_masks.append(masks[index])
            detections.mask = numpy.array(result_masks)
        return detections

    def __detect_objects_all(
        self,
        images_path: str,
        classes: list,
        box_threshold: float,
        text_threshold: float,
    ) -> ImageDetectionList:
        images_files = Path(images_path).glob("*")
        # load model
        grounding_dino_model = GDinoModel(
            model_config_path=self.gdino_config_path,
            model_checkpoint_path=self.gdino_ckpt_path,
            device=DEVICE,
        )
        detection_list = []

        for image_file in images_files:
            image = cv2.imread(str(image_file))

            image_detection = ImageDetection(
                file_name=str(image_file), image_array=image
            )
            image_detection.detections = grounding_dino_model.predict_with_classes(
                image=image,
                classes=classes,
                box_threshold=box_threshold,
                text_threshold=text_threshold,
            )
            detection_list.append(image_detection)

        # unload model
        del grounding_dino_model
        torch.cuda.empty_cache()

        return ImageDetectionList(
            detected_object=classes[0], detections_list=detection_list
        )

    def __segment_objects_all(
        self, detections_list: ImageDetectionList
    ) -> ImageDetectionList:
        # load model
        sam = sam_model_registry[self.sam_encoder_version](
            checkpoint=self.sam_ckpt_path
        )
        sam.to(device=DEVICE)
        sam_predictor = SamPredictor(sam)
        for image_detection in detections_list.detections_list:
            image_detection.detections = self.__segment_objects(
                sam_predictor=sam_predictor,
                image=image_detection.image_array,
                detections=image_detection.detections,
            )
        # unload model
        del sam
        torch.cuda.empty_cache()
        return detections_list

    def __reduce_bboxes_all(
        self, detection_list: ImageDetectionList
    ) -> ImageDetectionList:
        for image_detection in detection_list.detections_list:
            image_detection.detections = self.__reduce_bboxes(
                image_detection.detections
            )
        return detection_list

    def propose_region(
        self,
        images_path: str,
        prompt: list,
        box_threshold: float = 0.3,
        text_threshold: float = 0.25,
    ) -> ImageDetectionList:
        # detect object
        detections_list = self.__detect_objects_all(
            images_path=images_path,
            classes=prompt,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
        )
        # non_max supression(nms)
        detections_filtered_list = self.__reduce_bboxes_all(detections_list)
        # sam output
        detections_masks_list = self.__segment_objects_all(detections_filtered_list)
        # detection list
        return detections_masks_list
