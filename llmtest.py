from ollama import Client

client = Client()

# Ask a question
response = client.chat(
    model="llama3.2:3b",
    messages=[{"role": "user", "content": "What is RAG?"}],
    stream=False
)

print(response.message.content)
