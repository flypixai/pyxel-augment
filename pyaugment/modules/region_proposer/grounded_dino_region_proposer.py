import numpy
import torchvision
import torch
from pyaugment.modules.region_proposer.base_region_proposer import BaseRegionProposer
# TODO: Installing the next two is from source, running this necessiates 
# cloning grounded_segment_anything, is there a better way of doing this?
from groundingdino.util.inference import Model as GDinoModel
from segment_anything import sam_model_registry, SamPredictor
from supervision.detection.core import Detections


# TODO: find a better way to do this 
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class GroundedSAMRegionProposer(BaseRegionProposer):
    """
    A region proposer that combines object detection with object segmentation using two models: Grounding Dino and SAM.

    Parameters:
        gdino_config_path (str): The path to the GDino model configuration file.
        gdino_ckpt_path (str): The path to the GDino model checkpoint file.
        sam_encoder_version (str): The version of the SAM encoder to use.
        sam_ckpt_path (str): The path to the SAM model checkpoint file.

    Methods:
        propose_region(image, prompt, box_threshold=0.3, text_threshold=0.25):
            Detects objects in the input image using Grounding Dino , performs non-maximum suppression (NMS) on the detections,
            and segments the objects using SAM.

        __detect_objects__(image, classes, box_threshold, text_threshold):
            Detects objects in the input image using the Grounding Dino.

        __reduce_bboxes__(detections):
            Applies non-maximum suppression (NMS) to the given detections.

        __segment_objects__(image, xyxy):
            Segments objects in the input image using the SAM.

    Note:
        This class extends the BaseRegionProposer class and provides methods for proposing regions by
        combining object detection and segmentation using Grounding Dino and SAM.
    """
    def __init__(self, 
                 gdino_config_path, 
                 gdino_ckpt_path, 
                 sam_encoder_version,
                 sam_ckpt_path) -> None:
        super().__init__()
        self.gdino_config_path = gdino_config_path
        self.gdino_ckpt_path = gdino_ckpt_path
        self.sam_ckpt_path = sam_ckpt_path
        self.sam_encoder_version = sam_encoder_version
    def __reduce_bboxes__(self, detections: Detections) -> Detections:
        NMS_THRESHOLD = 0.8 ## TODO: find a better way to introduce this
        print(f"Before NMS: {len(detections.xyxy)} boxes")
        nms_idx = torchvision.ops.nms(
            torch.from_numpy(detections.xyxy), 
            torch.from_numpy(detections.confidence), 
            NMS_THRESHOLD
        ).numpy().tolist()

        detections.xyxy = detections.xyxy[nms_idx]
        detections.confidence = detections.confidence[nms_idx]
        detections.class_id = detections.class_id[nms_idx]

        print(f"After NMS: {len(detections.xyxy)} boxes")

        return detections
    def __detect_objects__(self, image: numpy.ndarray,
                       classes: str,
                       box_threshold: float,
                       text_threshold: float) -> Detections:
        # load model
        grounding_dino_model = GDinoModel(model_config_path=self.gdino_config_path, 
                                model_checkpoint_path=self.gdino_ckpt_path,
                                device=DEVICE)
        # detect
        detections = grounding_dino_model.predict_with_classes(
        image=image,
        classes=classes,
        box_threshold=box_threshold,
        text_threshold=text_threshold
        )
        # unload model
        del grounding_dino_model
        torch.cuda.empty_cache()
        
        return detections
    def __segment_objects__(self, image: numpy.ndarray,
                            xyxy: numpy.ndarray)-> list:
        # load model
        sam = sam_model_registry[self.sam_encoder_version](checkpoint=self.sam_ckpt_path)
        sam.to(device=DEVICE)
        sam_predictor = SamPredictor(sam)
        # segment
        sam_predictor.set_image(image)
        result_masks = []
        for box in xyxy:
            masks, scores, logits = sam_predictor.predict(
                box=box,
                multimask_output=True
            )
            index = numpy.argmax(scores)
            result_masks.append(masks[index])
        # unload model
        del sam
        torch.cuda.empty_cache()
        return result_masks
    def propose_region(self, image: numpy.ndarray, 
                       prompt: list,
                       box_threshold: float = 0.3,
                       text_threshold: float = 0.25) -> list:
        # detect object
        detections = self.__detect_objects__(
            image=image,
            classes=prompt,
            box_threshold=box_threshold,
            text_threshold=text_threshold)
        # non_max supression(nms)
        detections_filtered = self.__reduce_bboxes__(detections)
        # sam output
        detections_masks = self.__segment_objects__(
            image = image, 
            xyxy = detections_filtered.xyxy
        )
        return detections_masks