import gradio as gr
import torch
from PIL import Image
from hy3dgen.rembg import BackgroundRemover
from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline
import os
import tempfile

temp_dir = tempfile.gettempdir()
mesh_format = 'glb'

pipeline_mv = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
    'tencent/Hunyuan3D-2mv',
    subfolder='hunyuan3d-dit-v2-mv',
    variant='fp16'
)

pipeline_default = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
    'tencent/Hunyuan3D-2',
    subfolder='hunyuan3d-dit-v2-0',
    variant='fp16'
)

def process_image_test(font_image_path, left_image_path, back_image_path, seed):
    use_multiview = all([font_image_path, left_image_path, back_image_path])

    if use_multiview:
        images = {
            "front": font_image_path,
            "left": left_image_path,
            "back": back_image_path
        }
        for key in images:
            image = Image.open(images[key]).convert("RGBA")
            if image.mode == 'RGB':
                rembg = BackgroundRemover()
                image = rembg(image)
            images[key] = image

        mesh = pipeline_mv(
            image=images,
            num_inference_steps=50,
            octree_resolution=380,
            num_chunks=20000,
            generator=torch.manual_seed(seed),
            output_type='trimesh'
        )[0]
    else:
        image = Image.open(font_image_path).convert("RGBA")
        if image.mode == 'RGB':
            rembg = BackgroundRemover()
            image = rembg(image)

        mesh = pipeline_default(
            image=image,
            num_inference_steps=50,
            octree_resolution=380,
            num_chunks=20000,
            generator=torch.manual_seed(seed),
            output_type='trimesh'
        )[0]

    mesh_name = os.path.splitext(os.path.basename(font_image_path))[0]
    temp_mesh_path = os.path.join(temp_dir, f'{mesh_name}-{seed}.{mesh_format}')

    print(f"Exporting mesh to {temp_mesh_path}...")
    mesh.export(temp_mesh_path)
    return temp_mesh_path

with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column(scale=1):
            with gr.Row():
                front_image_input = gr.Image(type="filepath", label="Front Image", sources=["upload"])
            with gr.Row():
                left_image_input = gr.Image(type="filepath", label="Left Image", sources=["upload"])
                back_image_input = gr.Image(type="filepath", label="Back Image", sources=["upload"])
            with gr.Row():
                seed_input = gr.Slider(10000, 100000, step=1, label="Seed", value=10042)
            with gr.Row():
                process_button = gr.Button("Generate")
        with gr.Column(scale=4, min_width=600):
            model_output = gr.Model3D(label="3D Mesh Reconstruction", clear_color=[0.0, 0.0, 0.0, 0.0], height=800)

    process_button.click(
        fn = process_image_test,
        inputs = [front_image_input, left_image_input, back_image_input, seed_input],
        outputs = model_output
    )

demo.launch(inbrowser=True)
