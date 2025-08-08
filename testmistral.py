import requests

headers = {
    "Authorization": "Bearer <YOUR_API_KEY>",
    "Content-Type": "application/json"
}

data = {
    "model": "mistral-tiny",  # หรือ "mistral-small", "mistral-medium"
    "messages": [
        {"role": "user", "content": "อธิบาย Mistral API คืออะไร"}
    ]
}

response = requests.post(
    "https://api.mistral.ai/v1/chat/completions",
    headers=headers,
    json=data
)

print(response.json())
