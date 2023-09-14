from pyaugment.modules.region_proposer.grounded_dino_region_proposer import GroundedSAMRegionProposer
import os

# Specify the path to the directory you want to set as the current working directory
new_directory_path = os.path.expanduser('~/Grounded-Segment-Anything')

# Change the current working directory to the specified directory
os.chdir(new_directory_path)


proposer = GroundedSAMRegionProposer(gdino_config_path= "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
                                     gdino_ckpt_path = "./groundingdino_swint_ogc.pth",
                                     sam_encoder_version = "vit_h",
                                     sam_ckpt_path = "./sam_vit_h_4b8939.pth"
 
                                    )
images_path = "./assets_test"
prompt = ["The running dog"]
output = proposer.propose_region(images_path=images_path,
                                 prompt=prompt,
                                 box_threshold= 0.25,
                                 text_threshold= 0.25)