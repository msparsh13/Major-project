
import easyocr
from symspellpy import SymSpell, Verbosity
from PIL import Image, ImageOps
import numpy as np

def specialreadnews(img):
  # Initialize SymSpell
    sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
    sym_spell.load_dictionary("frequency_dictionary_en_82_765.txt", term_index=0, count_index=1)
    img_input = Image.fromarray(img)
    reader = easyocr.Reader(['en'])
    gray = ImageOps.grayscale(img_input)

    # Apply thresholding
    thresh = gray.point(lambda p: 255 if p > 150 else 0)
    img_np = np.array(thresh)

    results = reader.readtext(img_np, detail=0)
    corrected_results=[]
    for word in results:
      suggestions = sym_spell.lookup(word, Verbosity.CLOSEST, max_edit_distance=2)
      if suggestions:
        corrected_results.append(suggestions[0].term)
      else:
        corrected_results.append(word)

      
    return " ".join(corrected_results)
    