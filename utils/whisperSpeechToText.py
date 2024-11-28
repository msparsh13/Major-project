import whisper

# Load the Whisper model
model = whisper.load_model("tiny")  # Use "small", "medium", or "large" for better accuracy but higher resource usage.

def whisper_speech_to_text(audio_file):
    try:
        result = model.transcribe(audio_file)
        return result["text"]
    except Exception as e:
        return f"Error during transcription: {e}"
