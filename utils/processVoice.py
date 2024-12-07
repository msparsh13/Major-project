from functions.imageCaptioning import generate_captions
from functions.labelReading import specialreadnews
from functions.objectDetect import objectdetect
from .textToSpeech import text_to_speech
from utils.whisperSpeechToText import whisper_speech_to_text

classifyDict = ['classify', 'caption', 'explain', 'describe']
detectionDict = ['object', 'objects', 'detect', 'find']
readDict = ['read', 'label', 'written', 'transcribe', 'text']

def process_voice(audio, image):
    try:
        # Transcribe audio to text
        transcription = whisper_speech_to_text(audio).lower()
        print("transcription" , transcription)
        # text = classify(transcription)
        
        if any(word in transcription for word in classifyDict):
            result = generate_captions(image)
        elif any(word in transcription for word in detectionDict):
            result = objectdetect(image)
        elif any(word in transcription for word in readDict):
            result = specialreadnews(image)
        else:
            result = "Command not recognized. Please say 'caption', 'detect objects', or 'read labels'."
        
        print("result" ,result)
        # Convert result to audio
        audio_path = text_to_speech(result)
        return audio_path
    except Exception as e:
        error_message = f"Error processing voice input: {str(e)}"
        return text_to_speech(error_message)