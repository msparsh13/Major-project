import gradio as gr
from utils.processVoice import process_voice

with gr.Blocks() as demo:
    gr.Markdown("## NAYAN")
    
    # inputs
    with gr.Row():
        image_input = gr.Image(label="Capture Image") # later restrict to webcam only
        audio_input = gr.Audio(sources=['microphone'], type="filepath", label="Speak Your Command")
        
     # Output: Audio
    with gr.Row():
        output_audio = gr.Audio(label="Result Audio", type="filepath", autoplay=True)
        
    # Process button
    with gr.Row():
        process_btn = gr.Button("Process")
        
    process_btn.click(
        process_voice,
        inputs=[audio_input, image_input],
        outputs=output_audio,
    )

# Launch the interface
demo.launch()