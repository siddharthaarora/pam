
# ******************************************
# Call Jan's Local Chat API endpoint
# ******************************************
 
import requests

import requests

def call_jan_chat(text):
    url = 'http://localhost:1337/v1/chat/completions'
    headers = {
        'Content-Type': 'application/json'
    }
    data = {
        "messages": [
            {
            "content": "You are a helpful assistant.",
            "role": "system"
            },
            {
            "content": "Hello!",
            "role": "user"
            }
        ],
        "model": "llama3.2-3b-instruct",
        "stream": True,
        "max_tokens": 2048,
        "stop": [
            "hello"
        ],
        "frequency_penalty": 0,
        "presence_penalty": 0,
        "temperature": 0.7,
        "top_p": 0.95

    }

    response = requests.post(url, headers=headers, json=data)
    return response.text

# Example usage
response = call_jan_chat('Hello, how are you?')
print(response)




# ******************************************
# Call OpenAI Chat Completions endpoint
# ******************************************
# import os
# from openai import OpenAI

# llm = OpenAI(api_key="my OpenAI API key")

# system_prompt = """Given the following short description of a particular topic, 
#     write 3 attention grabbing headlines for a blog post. Reply with only titles, 
#     one on each line, with no additional text.
#     DESCRIPTION: 
# """

# user_input = """AI Orchestration woth LangChain and LlamaIndex
#     keywords: Generative AI, applications, LLM, chatbot"""

# response = llm.chat.completions.create(
#     model="gpt-3.5-turbo-0125",
#     max_tokens=500,
#     temperature=0.7,
#     messages=[
#         {"role": "system", "content": system_prompt},
#         {"role": "user", "content": user_input}
#     ]
# )

# print(response.choices[0].message.content)