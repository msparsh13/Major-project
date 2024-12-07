import whisper

# Load the Whisper model
model = whisper.load_model("small")  # Use "small", "medium", or "large" for better accuracy but higher resource usage.

def whisper_speech_to_text(audio_file):
    try:
        result = model.transcribe(audio_file, language='en')
        return result["text"]
    except Exception as e:
        return f"Error during transcription: {e}"
