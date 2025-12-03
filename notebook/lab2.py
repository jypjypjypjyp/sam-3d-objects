import os
import sys

sys.path.append("/home/yupeng.jia/Projects/worldmodel/thirdparty/sam3d")

import imageio
import uuid
from IPython.display import Image as ImageDisplay
# from inference import Inference, ready_gaussian_for_video_rendering, render_video, load_image, load_single_mask, display_image, make_scene, interactive_visualizer
from trellis.pipelines.inference_pipeline_trellis import InferencePipelineTrellis

PATH = "/home/yupeng.jia/Projects/worldmodel"
config_path = f"{PATH}/models/sam3d/pipeline.yaml"
import torch
torch.hub.set_dir(f"{PATH}/models/hub")
# inference = Inference(config_path, compile=False)
inference: InferencePipelineTrellis = InferencePipelineTrellis.from_pretrained("/home/yupeng.jia/Projects/worldmodel/models/trellis")
inference.cuda()
