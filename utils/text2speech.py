from gtts import gTTS
import tempfile

def text_to_speech(text):
    # Convert text to speech
    tts = gTTS(text)
    # Save to a temporary file
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3" )
    tts.save(temp_file.name)
    return text , temp_file.name

# Create Gradio interface
