import gradio as gr
from functions.labelReading import specialreadnews
from functions.imageCaptioning import generate_captions
from functions.objectDetect import objectdetect
from functions.labelReading import specialreadnews
from utils.processVoice import process_voice

with gr.Blocks() as demo:
    gr.Markdown("## NAYAN")
    
    # inputs
    with gr.Row():
        image_input = gr.Image(sources=['webcam'],label="Capture Image")
        audio_input = gr.Audio(sources=['microphone'], type="filepath", label="Speak Your Command")

    
    # Buttons for different functionalities
    # with gr.Row():
    #     btn_caption = gr.Button("Generate Captions")
    #     btn_detection = gr.Button("Object Detection")
    #     btn_labels = gr.Button("Read Labels")
    
    # Output display
    # with gr.Row():
    #     output_text = gr.Textbox(label="Output", lines=3)
        
     # Output: Audio
    with gr.Row():
        output_audio = gr.Audio(label="Result Audio", type="filepath")
        
    # Process button
    with gr.Row():
        process_btn = gr.Button("Process")

    # Button triggers
    # btn_caption.click(generate_captions, inputs=image_input, outputs=output_text)
    # btn_detection.click(objectdetect, inputs=image_input, outputs=output_text)
    # btn_labels.click(specialreadnews, inputs=image_input, outputs=output_text)
    
    process_btn.click(
        process_voice,
        inputs=[audio_input, image_input],
        outputs=output_audio,
    )


# Launch the interface
demo.launch()