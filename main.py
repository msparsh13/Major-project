import gradio as gr
from functions.labelReading import specialreadnews
# from functions.imageCaptioning import generate_captions
from functions.objectDetect import objectdetect
from functions.labelReading import specialreadnews
import os

   
with gr.Blocks() as demo:
    gr.Markdown("## NAYAN")
    
    # Camera input
    with gr.Row():
        image_input = gr.Image(label="Capture Image")
    
    # Buttons for different functionalities
    with gr.Row():
        btn_caption = gr.Button("Generate Captions")
        btn_detection = gr.Button("Object Detection")
        btn_labels = gr.Button("Read Labels")
    
    # Output display
    with gr.Row():
        output_text = gr.Textbox(label="Output", lines=3)

    # Button triggers
    # btn_caption.click(generate_captions, inputs=image_input, outputs=output_text)
    btn_detection.click(objectdetect, inputs=image_input, outputs=output_text)
    btn_labels.click(specialreadnews, inputs=image_input, outputs=output_text)

# Launch the interface
demo.launch()