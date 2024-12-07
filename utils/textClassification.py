# Load model directly
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_name = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name, torch_dtype="auto", device_map="auto")


def classify(text):
# Input text"
       try:
              prompt=f"""
Examples:
Text: find a laptop
Response: object detect

Text: what is in front of me
Response: image caption

Text: Explain the scene
Response: image caption

Text: read this
Response: read labels


based upon examples

Classify the following text into one of the given categories:
Categories: "image caption", "object detect", "read labels"

Text: {text}
Response:
"""
              inputs = tokenizer(prompt, return_tensors="pt") 
              outputs = model.generate(inputs["input_ids"] ,  max_length=50,  # Limit the length of the generated response
        temperature=0.5,  # Less randomness
        top_p=0.9    )
              print(outputs)
    # Decode and return the output
              response = tokenizer.decode(outputs[0], skip_special_tokens=True)
              print(response)
              return response.split("Response:")[-1].strip() 

       except Exception as e:
              print(e)
              return "None"
   
