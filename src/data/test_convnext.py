from transformers import AutoImageProcessor, ConvNextV2ForImageClassification
import torch
from datasets import load_dataset

dataset = load_dataset("huggingface/cats-image")
image = dataset["test"]["image"][0]

preprocessor = AutoImageProcessor.from_pretrained("facebook/convnextv2-large-1k-224")
model = ConvNextV2ForImageClassification.from_pretrained("facebook/convnextv2-large-1k-224")

inputs = preprocessor(image, return_tensors="pt")

with torch.no_grad():
    acts = model(**inputs)
    # Print the shapes of the activations
    print(acts)
