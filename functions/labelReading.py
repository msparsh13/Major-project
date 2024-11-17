import cv2
import easyocr

def specialreadnews(img):
    reader = easyocr.Reader(['en'])
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    result = reader.readtext(thresh, detail=0)
    txt=[]
    for (bbox, text, prob) in result:
        txt.append(text)
    return text
    