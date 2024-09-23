import openai
from openai import AzureOpenAI

client = AzureOpenAI()

# Define the text you want to translate and the target language
text_to_translate = "The approved law must still receive ratification from the upper house of parliament as well as approval by King Abdullah II, who retains supreme authority and whose signature is the seal of approval to all legislative matters."
target_language = "Kiswahili"  # Example: translating to French


completion = client.chat.completions.create(
    model="gpt-4o-2024-05-13",
    messages=[
        {
            "role": "user",
            "content": f"Translate the following text to {target_language}: {text_to_translate}",
        },
    ],
)
print(completion.choices[0].message.content)