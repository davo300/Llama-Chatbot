import ollama

# Use the model you actually have installed
model_name = "gemma3:1b"

response = ollama.chat(
    model=model_name,
    messages=[{"role": "user", "content": "Hello! Explain in 1 sentence why root 2 is irrational."}]
)

print("AI response:", response['message']['content'])
