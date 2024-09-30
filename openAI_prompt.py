import openai
import json
from openai import AzureOpenAI

client = AzureOpenAI()

input_texts = []
with open("input_dat.json", "r") as file:
    for line in file:
        input_texts.append(json.loads(line.strip()))
#input_data = json.load(file)


# Define the text you want to translate and the target language
translated_texts = []
for entry in input_texts:
    text_to_translate = entry["text"]
    target_language = "Kiswahili"

    #print(f"Processing: {text_to_translate}")

#text_to_translate = input_data["text"]

#text_to_translate = "The approved law must still receive ratification from the upper house of parliament as well as approval by King Abdullah II, who retains supreme authority and whose signature is the seal of approval to all legislative matters."
#target_language = "Kiswahili"  # Example: translating to French


completion = client.chat.completions.create(
    model="gpt-4o-2024-05-13",
    messages=[
        {
            "role": "user",
            "content": f"Translate the following text to {target_language}: {text_to_translate}",
        },
    ],
)

translated_text = completion.choices[0].message.content.strip()

#output_data = {
#    "original_text": text_to_translate,
#    "translated_text": translated_text,
#    "target_language": target_language
#}

translated_texts.append({
    "original_text": text_to_translate,
    "translated_text": translated_text,
    #"target_language": translated_text
    "language": "Kiswahili"
})

with open("output_data.json", "w", encoding="utf-8") as outfile:
    json.dump(translated_texts, outfile, ensure_ascii=False, indent=4)

print("Translation saved to output_data.json")

#print(completion.choices[0].message.content)
