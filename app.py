import gradio as gr
import torch
from PIL import Image
from hy3dgen.rembg import BackgroundRemover
from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline

pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
    'tencent/Hunyuan3D-2',
    subfolder='hunyuan3d-dit-v2-0',
    variant='fp16'
)

fast_pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
    'tencent/Hunyuan3D-2',
    subfolder='hunyuan3d-dit-v2-0-turbo',
    use_safetensors=True,
)
fast_pipeline.enable_flashvdm()

def process_image(image_path, seed):
    image = Image.open(image_path).convert("RGBA")
    if image.mode == 'RGB':
        rembg = BackgroundRemover()
        image = rembg(image)
    
    mesh = pipeline(
        image=image,
        num_inference_steps=50,
        octree_resolution=380,
        num_chunks=20000,
        generator=torch.manual_seed(seed),
        output_type='trimesh'
        )[0]
    
    temp_mesh_path = 'demo.glb'

    mesh.export(temp_mesh_path)
    return temp_mesh_path

preview_models = [gr.Model3D(), gr.Model3D(), gr.Model3D(), gr.Model3D()]

with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(type="filepath", label="Input Image")
            seed_input = gr.Slider(10000, 100000, step=1, label="Seed", value=10042)
            process_button = gr.Button("Generate")
        with gr.Column(scale=4, min_width=600):
            model_output = gr.Model3D(label="3D Mesh Reconstruction", clear_color=[0.0, 0.0, 0.0, 0.0], height=800)

    process_button.click(fn=process_image, inputs=[image_input, seed_input], outputs=model_output)

demo.launch(inbrowser=True)
