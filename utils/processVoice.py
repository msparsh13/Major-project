from functions.imageCaptioning import generate_captions
from functions.labelReading import specialreadnews
from functions.objectDetect import objectdetect
from utils.textClassification import classify
# from .speechToText import speech_to_text
from .textToSpeech import text_to_speech
from utils.whisperSpeechToText import whisper_speech_to_text

def process_voice(audio, image):
    try:
        # Transcribe audio to text
        transcription = whisper_speech_to_text(audio)
        # print("transcription" , transcription)
        text = classify(transcription)
        # print("text : " , text)
        # Determine which model to run:=> need to update this logic
        if "caption" in text or "caption" in transcription or "Describe" in transcription:
            result = generate_captions(image)
        elif "detect" in text or "object" in text or "Find" in transcription :
            result = objectdetect(image)
        elif "read" in text or "label" in text:
            result = specialreadnews(image)
        else:
            result = "Command not recognized. Please say 'caption', 'detect objects', or 'read labels'."
        
        # print("result" ,result)
        # Convert result to audio
        audio_path = text_to_speech(result)
        return audio_path
    except Exception as e:
        error_message = f"Error processing voice input: {str(e)}"
        return text_to_speech(error_message)