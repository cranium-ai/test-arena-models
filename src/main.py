from transformers import AutoModel

print("Loading models...")

model1 = AutoModel.from_pretrained("google-bert/bert-base-cased")
model2 = AutoModel.from_pretrained("google/gemma-2-27b")
model3 = AutoModel.from_pretrained("HuggingFaceTB/SmolLM2-1.7B")
model4 = AutoModel.from_pretrained("meta-llama/Llama-2-13b-chat-hf")
model5 = AutoModel.from_pretrained("microsoft/Phi-3.5-mini-instruct")
model6 = AutoModel.from_pretrained("nvidia/Llama-3.1-Nemotron-70B-Instruct-HF")
model8 = AutoModel.from_pretrained("meta-llama/Llama-3.2-1B")

print("Models loaded successfully!")
