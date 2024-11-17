import easyocr

def specialreadnews(img):
  try:
    reader = easyocr.Reader(['en'])  # Specify the languages to detect
    results = reader.readtext(img, detail=0)
    return " ".join(results) if results else "No text found in the image."
  except Exception as e:
    return f"Error during OCR: {str(e)}"
    