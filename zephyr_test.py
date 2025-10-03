import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load model + tokenizer
model_id = "HuggingFaceH4/zephyr-7b-beta"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

# Create a test prompt
prompt = "Write a short story about a robot learning to paint."

# Tokenize
inputs = tokenizer(prompt, return_tensors="pt")

# Generate
outputs = model.generate(
    **inputs,
    max_new_tokens=50,   # more room for output
    do_sample=True,
    temperature=0.7,
)

# Decode full text
full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

# Extract continuation only
prompt_len = inputs["input_ids"].shape[1]
generated_text = tokenizer.decode(outputs[0][prompt_len:], skip_special_tokens=True)

# Print
print("=== MODEL OUTPUT ===")
print(generated_text.strip())
