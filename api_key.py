import openai

openai.api_key = "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"

response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": "Hello, how are you?"}]
)

print(response["choices"][0]["message"]["content"])
