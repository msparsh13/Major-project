import gradio as gr
from functions.labelReading import specialreadnews
from functions.imageCaptioning import generate_captions
from functions.objectDetect import objectdetect
from utils.text2speech import text_to_speech
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
    
    # Button triggers
    with gr.Row():
        output_text = gr.Textbox(label="Output", lines=3)
        output_audio = gr.Audio(label="Audio Output" , autoplay=True)
    
    # Button triggers with TTS integration
    btn_caption.click(lambda img: text_to_speech(generate_captions(img)), 
                     inputs=image_input, outputs=[output_text, output_audio])
    
    btn_detection.click(lambda img: text_to_speech(objectdetect(img)), 
                        inputs=image_input, outputs=[output_text, output_audio])
    
    btn_labels.click(lambda img: text_to_speech(specialreadnews(img)), 
                     inputs=image_input, outputs=[output_text, output_audio])

# Launch the interface
demo.launch()