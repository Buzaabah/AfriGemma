import openai
from openai import AzureOpenAI

client = AzureOpenAI()

from openai import AzureOpenAI

client = AzureOpenAI()

completion = client.chat.completions.create(
    model="gpt-4o-2024-05-13",
    messages=[
        {
            "role": "user",
            "content": "How do I output all files in a directory using Python?",
        },
    ],
)
print(completion.to_json())