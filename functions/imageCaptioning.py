from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from utils.textToSpeech import text_to_speech

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

def generate_captions(img):
    img_input = Image.fromarray(img)
    inputs = processor(img_input, return_tensors="pt")
    out = model.generate(**inputs)
    caption = processor.decode(out[0], skip_special_tokens=True)
    filePath = text_to_speech(caption)
    return filePath
