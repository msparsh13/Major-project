from vosk import Model, KaldiRecognizer

model = Model("../models/Vosk/vosk-model-small-hi-0.22")
recognizer = KaldiRecognizer(model, 16000)

def voiceToText(audio_stream):
    data = audio_stream.read(4096, exception_on_overflow=False)
    
    if recognizer.AcceptWaveform(data):
        result = recognizer.Result()
        text = result[14:-3]  
        return text
    return None 
