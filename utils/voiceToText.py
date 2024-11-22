from vosk import Model, KaldiRecognizer
import pyaudio

model = Model("./Vosk/vosk-model-small-hi-0.22")

recognizer = KaldiRecognizer(model, 16000)

mic = pyaudio.PyAudio()

def textToVoice(audio_stream):
    data = audio_stream.read(4096, exception_on_overflow=False)
    
    if recognizer.AcceptWaveform(data):
        result = recognizer.Result()
        text = result[14:-3]  
        return text
    return None 
