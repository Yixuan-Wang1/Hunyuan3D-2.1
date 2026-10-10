import os
import time
import torch
from PIL import Image

from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline


ROOT = "/data/yixuan.wang/Hunyuan3D/Hunyuan3D-2.1"
MODEL_DIR = os.path.join(ROOT, "models")
INPUT_IMAGE = "/data/yixuan.wang/TACO/dataset/Egocentric_RGB_Images_Masks/(brush, brush, bowl)/20230927_027/000001/000001_tool_cutout_tight.png"
OUTPUT_DIR = "/data/yixuan.wang/TACO/dataset/Recon_Object_Models/(brush, brush, bowl)/20230927_027/000001"
OUTPUT_MESH = os.path.join(OUTPUT_DIR, "tool.obj")

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Loading image:", INPUT_IMAGE)
image = Image.open(INPUT_IMAGE).convert("RGBA")

print("Loading Hunyuan3D Shape pipeline...")
pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
    MODEL_DIR,
    subfolder="hunyuan3d-dit-v2-1",
    use_safetensors=False,
    device="cuda",
)

generator = torch.Generator(device="cuda").manual_seed(12345)

print("Generating geometry...")
start_time = time.time()

meshes = pipeline(
    image=image,
    num_inference_steps=30,
    guidance_scale=5.0,
    generator=generator,
)

mesh = meshes[0]
mesh.export(OUTPUT_MESH)

print("Saved mesh:", OUTPUT_MESH)
print("Elapsed seconds:", time.time() - start_time)