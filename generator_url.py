from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import requests

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Test with image URL
url = "https://share.google/rp17WwxHZP0gTpZMf"
image = Image.open(requests.get(url, stream=True).raw)

inputs = processor(image, return_tensors="pt")
output = model.generate(**inputs)
caption = processor.decode(output[0], skip_special_tokens=True)

print("Caption:", caption)
