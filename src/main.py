from transformers import AutoModel

print("Loading models...")

model3 = AutoModel.from_pretrained("HuggingFaceTB/SmolLM2-135M")

print("Models loaded successfully!")

