import openai

openai.api_base = "http://localhost:8091/v1"
openai.api_key = ""
prompt = "What is the capital of Ohio?"
messages = [{"role": "system", "content": prompt}]

response = openai.ChatCompletion.create(
    model="Mistral-7B-OpenOrca",
    messages=messages,
    temperature=1.31,
    max_tokens=8192,
    top_p=1.0,
    n=1,
    stream=False,
)
print(response)