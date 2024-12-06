from functions.imageCaptioning import generate_captions
from functions.labelReading import specialreadnews
from functions.objectDetect import objectdetect
from utils.textToSpeech import text_to_speech
from utils.whisperSpeechToText import whisper_speech_to_text

def process_voice(audio, image):
    try:
        # Transcribe audio to text
        transcription = whisper_speech_to_text(audio)

        # Determine which model to run:=> need to update this logic
        if "caption" in transcription.lower():
            result = generate_captions(image)
        elif "detect" in transcription.lower() or "objects" in transcription.lower():
            result = objectdetect(image)[0]
        elif "read" in transcription.lower() or "labels" in transcription.lower():
            result = specialreadnews(image)
        else:
            result = "Command not recognized. Please say 'caption', 'detect objects', or 'read labels'."
        
        # Convert result to audio
        audio_path = text_to_speech(result)
        return audio_path
    except Exception as e:
        error_message = f"Error processing voice input: {str(e)}"
        return text_to_speech(error_message)