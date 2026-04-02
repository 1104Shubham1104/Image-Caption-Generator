from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import sys
import torch

if len(sys.argv) < 2:
    print("Usage: python generator_local.py <image_path>")
    exit()

image_path = sys.argv[1]

# Load model + processor
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Load image
image = Image.open(image_path)

# Encode inputs
inputs = processor(image, return_tensors="pt")

# Generate 5 captions
num_captions = 5
captions = []

for _ in range(num_captions):
    output = model.generate(
        **inputs,
        max_length=30,
        num_beams=1,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=1.1
    )
    caption = processor.decode(output[0], skip_special_tokens=True)
    captions.append(caption)

# Remove duplicates
unique_captions = list(dict.fromkeys(captions))

print("\nGenerated captions:")
for i, c in enumerate(unique_captions, 1):
    print(f"{i}. {c}")
