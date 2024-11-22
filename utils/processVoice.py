from functions.imageCaptioning import generate_captions
from functions.labelReading import specialreadnews
from functions.objectDetect import objectdetect
from .speecToText import speech_to_text
from .textToSpeech import text_to_speech

def process_voice(audio, image):
    """   
    Process voice input, decide the model to use, and return results as audio.
    """
    try:
        # Transcribe audio to text
        transcription = speech_to_text(audio)
        print(transcription)

        # Determine which model to run
        if "caption" in transcription:
            result = generate_captions(image)
        elif "detect" in transcription or "objects" in transcription:
            result = objectdetect(image)
        elif "read" in transcription or "labels" in transcription:
            result = specialreadnews(image)
        else:
            result = "Command not recognized. Please say 'caption', 'detect objects', or 'read labels'."
        
        # Convert result to audio
        audio_path = text_to_speech(result)
        return audio_path
    except Exception as e:
        error_message = f"Error processing voice input: {str(e)}"
        return text_to_speech(error_message)