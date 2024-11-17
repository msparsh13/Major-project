import cv2
import easyocr
from symspellpy import SymSpell, Verbosity
from textblob import TextBlob


def specialreadnews(img):
  # Initialize SymSpell
    sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
    sym_spell.load_dictionary("frequency_dictionary_en_82_765.txt", term_index=0, count_index=1)
    image = cv2.imread(img)
    reader = easyocr.Reader(['en'])
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

    results = reader.readtext(thresh, detail=0)
    corrected_results=[]
    for word in results:
      suggestions = sym_spell.lookup(word, Verbosity.CLOSEST, max_edit_distance=2)
      if suggestions:
        corrected_results.append(suggestions[0].term)
      else:
        corrected_results.append(word)

      
    return " ".join(corrected_results)
    