import os
import argparse
import os
import glob
import uuid
# os.environ['ATTN_BACKEND'] = 'xformers'   # Can be 'flash-attn' or 'xformers', default is 'flash-attn'
os.environ['SPCONV_ALGO'] = 'native'        # Can be 'native' or 'auto', default is 'auto'.
                                            # 'auto' is faster but will do benchmarking at the beginning.
                                            # Recommended to set to 'native' if run only once.

import numpy as np
import imageio
import logging
from PIL import Image
from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.utils import render_utils, postprocessing_utils,general_utils
import open3d as o3d
import sys
import traceback
# import bpy    
import pymeshlab
import pymeshlab.pmeshlab



# Set up logging
logging.basicConfig(
    level=logging.INFO,  # Adjust the level as needed (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger()

def main():
    parser = argparse.ArgumentParser(
        description="使用 rembg 去除指定文件夹下所有图片的背景"
    )
    parser.add_argument(
        '--input', '-i', required=True,
        help="输入文件夹路径，包含待处理图片"
    )
    parser.add_argument(
        '--output', '-o', default="./output",
        help="输出文件夹路径，处理后的图片将保存于此"
    )
    
    args = parser.parse_args()
    # 生成随机文件夹后缀
    random_suffix = str(uuid.uuid4())[:8]  # 取UUID的前8位作为后缀

    # 构造最终的输出路径
    final_output_path = os.path.join(args.output, random_suffix)

    # 确保输出目录存在
    os.makedirs(final_output_path, exist_ok=True)
    images_output_path = os.path.join(final_output_path, "images_RMBG")
    if not os.path.exists(images_output_path):
        os.makedirs(images_output_path)
    # Load a pipeline from a model folder or a Hugging Face model hub.
    
    pipeline = TrellisImageTo3DPipeline.from_pretrained("./ckpts/TRELLIS-image-large")
    pipeline.cuda()
    # 获取输入文件夹下的所有图片（假设是 JPG/PNG 格式）
    image_list = sorted(glob.glob(os.path.join(args.input, "*.*")))  # 匹配所有文件

    # 仅保留图片格式（可扩展）
    image_list = [img for img in image_list if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    # Load an image
    # images = [
    #     Image.open("assets/example_multi_image/character_1.png"),
    #     Image.open("assets/example_multi_image/character_2.png"),
    #     Image.open("assets/example_multi_image/character_3.png"),
    # ]
    # 加载图片
    images = [Image.open(img) for img in image_list]

    # Run the pipeline
    # outputs = pipeline.run_multi_image(
    #     images,
    #     seed=1,
    #     # Optional parameters
    #     sparse_structure_sampler_params={
    #         "steps": 12,
    #         "cfg_strength": 7.5,
    #     },
    #     slat_sampler_params={
    #         "steps": 12,
    #         "cfg_strength": 3,
    #     },
    # )
    outputs,images_RMBG = pipeline.run_multi_image_v2(
        images,
        seed=1,
        # Optional parameters
        sparse_structure_sampler_params={
            "steps": 100,
            "cfg_strength": 7.5,
        },
        slat_sampler_params={
            "steps": 100,
            "cfg_strength": 3,
        },
    )
    # outputs is a dictionary containing generated 3D assets in different formats:
    # - outputs['gaussian']: a list of 3D Gaussians
    # - outputs['radiance_field']: a list of radiance fields
    # - outputs['mesh']: a list of meshes

    general_utils.save_imgs(images_RMBG,images_output_path)
    
    video_gs = render_utils.render_video(outputs['gaussian'][0])['color']
    video_mesh = render_utils.render_video(outputs['mesh'][0])['normal']
    video = [np.concatenate([frame_gs, frame_mesh], axis=1) for frame_gs, frame_mesh in zip(video_gs, video_mesh)]
    imageio.mimsave(os.path.join(final_output_path, "sample_multi.mp4"), video, fps=30)
    # GLB files can be extracted from the outputs
    
    simplify=0#0.95
    texture_size=2048#1024
    glb = postprocessing_utils.to_glb(
        outputs['gaussian'][0],
        outputs['mesh'][0],
        # Optional parameters
        simplify=simplify,          # Ratio of triangles to remove in the simplification process
        texture_size=texture_size,      # Size of the texture used for the GLB
    )
    glb.export(os.path.join(final_output_path, "sample.glb"))

    # Save Gaussians as PLY files
    outputs['gaussian'][0].save_ply(os.path.join(final_output_path, "sample.ply"))
    
    logger.info(f"Process completed. Files saved at {final_output_path}")
    #o3d.io.write_triangle_mesh(os.path.join(args.output, "sample.obj"), outputs['mesh'][0])
    
if __name__ == "__main__":
    main()