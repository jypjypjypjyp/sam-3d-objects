import os
import sys

sys.path.append("/home/yupeng.jia/Projects/worldmodel/thirdparty/sam3d")

import imageio
import uuid
from IPython.display import Image as ImageDisplay
from inference import Inference, ready_gaussian_for_video_rendering, render_video, load_image, load_single_mask, display_image, make_scene, interactive_visualizer


PATH = "/home/yupeng.jia/Projects/worldmodel"
config_path = f"{PATH}/models/sam3d/pipeline.yaml"
import torch
torch.hub.set_dir(f"{PATH}/models/hub")
inference = Inference(config_path, compile=False)

IMAGE_PATH = f"{PATH}/thirdparty/sam3d/notebook/images/shutterstock_stylish_kidsroom_1640806567/image.png"
IMAGE_NAME = os.path.basename(os.path.dirname(IMAGE_PATH))

image = load_image(IMAGE_PATH)
mask = load_single_mask(os.path.dirname(IMAGE_PATH), index=14)
# display_image(image, masks=[mask])

# run model
output = inference(image, mask, seed=42)
output["glb"].export("a.glb")

# export gaussian splat (as point cloud)
# output["gs"].save_ply(f"{PATH}/thirdparty/sam3d/notebook/gaussians/single/{IMAGE_NAME}.ply")

# render gaussian splat
# scene_gs = make_scene(output)
# scene_gs = ready_gaussian_for_video_rendering(scene_gs)

# video = render_video(
#     scene_gs,
#     r=1,
#     fov=60,
#     pitch_deg=15,
#     yaw_start_deg=-45,
#     resolution=512,
# )["color"]

# # save video as gif
# imageio.mimsave(
#     os.path.join(f"{PATH}/thirdparty/sam3d/notebook/gaussians/single/{IMAGE_NAME}.gif"),
#     video,
#     format="GIF",
#     duration=1000 / 30,  # default assuming 30fps from the input MP4
#     loop=0,  # 0 means loop indefinitely
# )

# notebook display
# ImageDisplay(url=f"gaussians/single/{IMAGE_NAME}.gif?cache_invalidator={uuid.uuid4()}")