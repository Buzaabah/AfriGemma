import openai
import json
from openai import AzureOpenAI

client = AzureOpenAI()

start_index = 1895
# Load input JSON lines
input_texts = []

with open("input_dat.json", "r") as file:
    for i, line in enumerate(file):
        if i >= start_index:
            input_texts.append(json.loads(line.strip()))

target_language = "Kiswahili"
#translated_texts = []

# Translate each text entry
for entry in input_texts:
    if "text" in entry:
        text_to_translate = entry["text"]

        try:
            completion = client.chat.completions.create(
                model="gpt-4o-2024-05-13",
                messages=[
                    {
                        "role": "user",
                        "content": f"Translate the following text to {target_language}: {text_to_translate}",
                    },
                ],
            )

            if completion and completion.choices:
                translated_text = completion.choices[0].message.content.strip()
                #entry["translated_text"] = translated_text
                #entry["language"] = target_language
                #translated_texts.append(entry)

                with open("Only_translated.jsonl", "a", encoding="utf-8") as outfile:
                    #json.dump(entry, outfile, ensure_ascii=False)
                    json.dump({"translated_text": translated_text}, outfile, ensure_ascii=False)
                    outfile.write("\n")
            else:
                print(f"Warning: No valid response for text: {text_to_translate}")

        except Exception as e:
            print(f"Error translating text: {text_to_translate}")
            print(f"Exception: {e}")

# Save translated entries to a new JSONL file
#with open("translated.jsonl", "w", encoding="utf-8") as outfile:
#    for translated_entry in translated_texts:
#        json.dump(translated_entry, outfile, ensure_ascii=False)
#        outfile.write("\n")

print("Translation saved to translated.jsonl")



