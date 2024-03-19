import json
import argparse
import sys
from llama_cpp import Llama

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", type=str, default="models/mistral-7b-openorca/mistral-7b-openorca.Q4_K_M.gguf")
args = parser.parse_args()

llm = Llama(model_path=args.model)

prompt = """<|im_start|>system

  You are a helpful assistant.<|im_end|>

  <|im_start|>user

  Write a 500 words essay about Vietnam?<|im_end|>

  <|im_start|>assistant

"""

stream = llm(
    prompt,
    max_tokens=1024,
    stop=["Q:", "\n"],
    stream=False,
)
print(stream)
# for chunk in stream:
    # print(chunk)




# import openai

# openai.api_base = "http://localhost:8091/v1"
# openai.api_key = ""
# prompt = "Write a 500-word essay about Vietnam?"
# messages = [{"role": "system", "content": prompt}]

# response = openai.ChatCompletion.create(
#     model="Mistral-7B-OpenOrca",
#     messages=messages,
#     temperature=1.31,
#     max_tokens=819,
#     top_p=1.0,
#     n=1,
#     stream=True,
# )

# # print(response)

# for chunk in response:
#     print(chunk.choices[0].text)
#     # sys.stdout.write(chunk.choices[0].text)
#     # sys.stdout.flush()