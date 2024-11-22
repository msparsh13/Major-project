from functions.imageCaptioning import generate_captions
from functions.labelReading import specialreadnews
from functions.objectDetect import objectdetect
from .voiceToText import voiceToText
from .textToVoice import textToVoice

def process_voice(audio, image):
    """   
    Process voice input, decide the model to use, and return results as audio.
    """
    try:
        # Transcribe audio to text
        transcription = voiceToText(audio)

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
        audio_path = textToVoice(result)
        return audio_path
    except Exception as e:
        error_message = f"Error processing voice input: {str(e)}"
        return textToVoice(error_message)