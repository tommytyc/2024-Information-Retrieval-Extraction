import json
import os
from dotenv import load_dotenv
from openai import OpenAI, RateLimitError
from tqdm import tqdm
from itertools import cycle
load_dotenv()

LLM_MODEL = "Meta-Llama-3.3-70B-Instruct"
clients = [OpenAI(api_key=os.getenv(f"SAMBANOVA_API_KEY{k}"), base_url="https://api.sambanova.ai/v1") for k in range(1, 6)]
clients = cycle(clients)
client = next(clients)

def load_train_test_dialogues():
    with open('data/train.jsonl', 'r') as f:
        train_dialogues = [json.loads(line) for line in f]
    with open('data/test.jsonl', 'r') as f:
        test_dialogues = [json.loads(line) for line in f]

    return train_dialogues, test_dialogues

def generate_image_descriptor(dialogue_history):
    global client
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant that generates descriptions of images based on dialogue context.",
        },
        {
            "role": "user",
            "content": f"""Please read the following dialogue context: {" ".join([i['message'] for i in dialogue_history])}
            Based on the dialogue context, please describe
            the photograph shared by speaker A.
            List the answer in JSON format.
            - main subject: {{simply list the answer by ','}}
            - prominent objects in the foreground: {{simply list the answer by ','}}
            - background scene: {{one background scene}}
            - events: {{simply list the answer by ','}}
            - materials and attributes: {{simply list the answer by ','}}
            Answers:""",
        },
    ]
    try:
        response = client.chat.completions.create(
            model=LLM_MODEL, messages=messages, temperature=0.1, max_tokens=100
        )
    except RateLimitError:
        client = next(clients)
        response = client.chat.completions.create(
            model=LLM_MODEL, messages=messages, temperature=0.1, max_tokens=100
        )
    return response.choices[0].message.content

def main():
    train_dialogues, test_dialogues = load_train_test_dialogues()
    for dialogue in tqdm(train_dialogues):
        dialogue_history = dialogue['dialogue']
        image_descriptor = generate_image_descriptor(dialogue_history).replace('```json\n', '').replace('\n```', '').strip()
        dialogue['image_descriptor'] = image_descriptor

    with open('data/train_img_desc.jsonl', 'w') as f:
        for dialogue in train_dialogues:
            f.write(json.dumps(dialogue) + '\n')

    for dialogue in tqdm(test_dialogues):
        dialogue_history = dialogue['dialogue']
        image_descriptor = generate_image_descriptor(dialogue_history).replace('```json\n', '').replace('\n```', '').strip()
        dialogue['image_descriptor'] = image_descriptor

    with open('data/test_img_desc.jsonl', 'w') as f:
        for dialogue in test_dialogues:
            f.write(json.dumps(dialogue) + '\n')

if __name__ == '__main__':
    main()