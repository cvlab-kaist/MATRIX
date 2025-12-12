'''
    Generate mask for image using SAM2 with User prompt.

    1. Select an image from the folder.
    2. Click Image to provide point prompts or bbox prompts.
    3. Click 'Run' to generate mask for the selected image.
    4. Give object id to the object in the image.
    5. Click 'Save' to save the mask and visualization image.
'''
import os
import argparse
import gradio as gr
import imageio.v2 as iio
import numpy as np

from core_img import CoreImg
from utils import visualize_mask, visualize_points

core = CoreImg()

def run(args):
    image_files = [f for f in os.listdir(args.image_dir) 
                   if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
    image_files.sort()

    global core
    core.set_config(
        sam2_model_id=args.sam2_model_id,
        image_dir=args.image_dir,
        output_dir=args.output_dir,
    )
    core.initialize()
    
    with gr.Blocks() as demo:
        with gr.Column():
            with gr.Row():
                image_id = gr.Dropdown(
                    choices=image_files,
                    label="select image",
                )
                point_type = gr.Radio(["positive", "negative"], label="Select point type")
                obj_id = gr.Number(value=0, label="Object ID")
            with gr.Row():
                selected_image = gr.Image(
                    type="numpy", 
                    label="selected image", 
                    show_label=False, 
                    show_download_button=False
                )
                masked_image = gr.Image(
                    type="numpy",
                    label="segmented result",
                    show_label=False,
                    show_download_button=False
                )
            
            with gr.Row():
                with gr.Row():
                    undo_button = gr.Button(
                        value="Undo",
                    )
                    run_button = gr.Button(
                        value="Run Segmentation",
                    )
                save_button = gr.Button(
                    value="Save",
                )
        
        # 1. Select image
        image_id.change(
            fn=select_image,
            inputs=[image_id],
            outputs=[selected_image, masked_image],
        )

        # 2. Generate mask for the selected image
        selected_image.select(
            fn=get_point,
            inputs=[point_type],
            outputs=[selected_image],
        )

        # 3. Reset prompts
        undo_button.click(
            fn=undo_point,
            inputs=[],
            outputs=[selected_image],
        )

        # 4. Run segmentation
        run_button.click(
            fn=run_segmentation,
            inputs=[],
            outputs=[masked_image],
        )

        # 5. Save the mask and visualization image
        save_button.click(
            fn=save_outputs,
            inputs=[obj_id],
            outputs=[],
        )

    demo.queue().launch(
        debug=True, 
        server_port=args.port,
        allowed_paths=[args.image_dir, args.output_dir],
    )

def select_image(image_id):
    global core
    print(f"SELECT IMAGE - IMAGE ID: {image_id}")
    if image_id is None:
        return None, None
    
    core.set_image(image_id)
    core.init_state()
    selected_image = core.get_selected_image()
    return selected_image, None

def get_point(point_type, evt: gr.SelectData):
    global core
    print(f"GET POINT - POINT TYPE: {point_type}, EVENT INDEX: {evt.index}, TOTAL POINTS: {len(core.points)}")
    core.get_point(
        point_coords=evt.index,
        point_type=point_type,
    )
    selected_image = core.get_selected_image()
    selected_image = visualize_points(selected_image, core.points)
    return selected_image

def undo_point():
    global core
    print(f"UNDO POINT - TOTAL POINTS: {len(core.points) - 1}")
    core.undo_point()
    selected_image = core.get_selected_image()
    selected_image = visualize_points(selected_image, core.points)
    return selected_image

def run_segmentation():
    global core
    print("RUN SEGMENTATION")
    mask = core.run()
    if mask is not None:
        selected_image = core.get_selected_image()
        masked_image = visualize_mask(selected_image, mask)
        core.current_mask = mask
        return masked_image
    return None

def save_outputs(obj_id):
    global core
    if hasattr(core, 'current_mask') and core.current_mask is not None:
        core.save_outputs(obj_id)
        print(f"SAVE OUTPUTS - OBJECT ID: {obj_id}")
    else:
        print("No mask to save. Please run segmentation first.")

def parse_args():
    parser = argparse.ArgumentParser()
    # DATA
    parser.add_argument(
        "--sam2_model_id",
        type=str,
        default="facebook/sam2.1-hiera-large",
        help="Model ID for the SAM2 model to be used.",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/path/to/data",
        help="Directory containing image files for processing.",
    )
    parser.add_argument(
        "--image_dir",
        type=str,
        default="/path/to/image_dir",
        help="Directory containing image files for processing.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/path/to/output_dir",
        help="Directory to save processed output files.",
    )
    # ENV
    parser.add_argument(
        "--port",
        type=int,
        default=7861,
        help="Port number for the Gradio interface.",
    )
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    
    run(args)





