import json
from datasets import Dataset as dDataset, DatasetDict
from tqdm import tqdm
import os
from dotenv import load_dotenv
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoProcessor, AutoTokenizer, AutoModelForZeroShotImageClassification
from PIL import Image
import requests

load_dotenv()

# --- Configuration ---
# LLM_MODEL = "grok-2-1212"
# VLM_MODEL = "Salesforce/blip2-opt-2.7b"
# VLM_MODEL = "openai/clip-vit-base-patch32"
VLM_MODEL = "openai/clip-vit-large-patch14"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TRAIN_DATA_FILE = "data/train_img_desc.jsonl"
# TRAIN_DATA_FILE = "data/train_data.jsonl"
# DEV_DATA_FILE = "data/dev_data.jsonl"
TRAIN_IMAGE_DIR = "data/train_images/train_images/"
TEST_DATA_FILE = "data/test_img_desc.jsonl"
TEST_IMAGE_DIR = "data/test_images/test_images/"
TRAIN_DATASET_SAVE_PATH = "./data/train_dataset"
TEST_DATASET_SAVE_PATH = "./data/test_dataset"
CHECKPOINT_DIR = "./checkpoints"
EPOCHS = 15
BATCH_SIZE = 64
LEARNING_RATE = 5e-5
TEMP = 0.5

class ImageDataset(Dataset):
    def __init__(self, data, processor, tokenizer):
        self.data = data
        self.processor = processor
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        images = load_image_from_path(os.path.join('data/train_images/train_images', self.data[idx]['photo_path']))
        query = self.data[idx]["image_descriptor"].split("{")[1].split("}")[0].replace("}", "").replace("\n", "").strip()
        images = self.processor(images=images, return_tensors="pt", do_center_crop=True, do_resize=True)['pixel_values']
        query = self.tokenizer(query, return_tensors="pt", padding="max_length", max_length=77)
        # query = self.processor(text=query, return_tensors="pt")
        query = {k: v[:, :77] for k, v in query.items()}
        return images, query

# --- LLM (Descriptor Generation) ---
# client = OpenAI(api_key=os.getenv("XAI_API_KEY"), base_url="https://api.x.ai/v1/")

def load_dialogue_data(filepath):
    """Loads dialogue data from a JSONL file."""
    data = []
    with open(filepath, "r") as f:
        for line in f:
            data.append(json.loads(line))
    return data

def load_vlm(model_name, device, checkpoint_dir=None):
    if checkpoint_dir is not None:
        model = AutoModelForZeroShotImageClassification.from_pretrained(checkpoint_dir, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
    else:
        model = AutoModelForZeroShotImageClassification.from_pretrained(model_name, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
    processor = AutoProcessor.from_pretrained(model_name)
    model.to(device)
    return model, processor

def load_image_from_path(image_path):
    """Loads an image, handling both local paths and URLs."""
    if image_path.startswith("http"):
        image = Image.open(requests.get(image_path, stream=True).raw)
    else:
        image = Image.open(image_path)
    return image

def contrastive_loss(q, p):
    batch_size = q.size(0)
    mask = torch.eye(batch_size, dtype=torch.bool).to(DEVICE)
    q_norm = q / q.norm(dim=1)[:, None]
    p_norm = p / p.norm(dim=1)[:, None]
    sim_mat = torch.mm(q_norm, p_norm.transpose(0,1))
    nominator = (mask * torch.exp(sim_mat / TEMP)).sum(dim=1)
    denominator = (~mask * torch.exp(sim_mat / TEMP)).sum(dim=1) + 1e-6 + nominator
    all_losses = -torch.log(nominator / denominator)
    loss = torch.sum(all_losses) / batch_size
    return loss

def train_clip_model(dialogue_dataset):
    # Load the model
    model, processor = load_vlm(VLM_MODEL, DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(VLM_MODEL)
    dataset = ImageDataset(dialogue_dataset, processor, tokenizer)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    for epoch in range(EPOCHS):
        for batch in tqdm(dataloader, desc=f"Training Epoch {epoch}"):
            images, queries = batch
            queries = {k: v.squeeze().to(DEVICE) for k, v in queries.items()}
            image_embeddings = model.get_image_features(images.squeeze().to(DEVICE))
            text_embeddings = model.get_text_features(**queries)
            loss = contrastive_loss(image_embeddings, text_embeddings)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch} Loss: {loss.item()}")
    model.save_pretrained(CHECKPOINT_DIR)
    return model

def search(dataset: Dataset, query: str, k: int = 30):
    """a function that embeds query text and returns the most probable image"""
    model, processor = load_vlm(VLM_MODEL, DEVICE, CHECKPOINT_DIR)
    # tokenizer = AutoTokenizer.from_pretrained(VLM_MODEL)
    inputs = processor(text=query, return_tensors="pt").to(DEVICE)
    inputs = {k: v[:, :77] for k, v in inputs.items()}
    # inputs = tokenizer(query, return_tensors="pt", padding=True).to(DEVICE)
    text_embeddings = model.get_text_features(**inputs).detach().to(torch.float).cpu().numpy()
    retrieved_examples = dataset.get_nearest_examples("embeddings", text_embeddings, k=k)
    return retrieved_examples

def prepare_img_db(image_dir, dataset_type='train'):
    model, processor = load_vlm(VLM_MODEL, DEVICE, CHECKPOINT_DIR)
    image_files = os.listdir(image_dir)
    image_files = [os.path.join(image_dir, img) for img in image_files]

    dataset_data = []
    model.eval()
    with torch.no_grad():
        for image_path in tqdm(image_files, desc=f"Processing {dataset_type} Images"):
            image = load_image_from_path(image_path)
            inputs = processor(images=image, return_tensors="pt", do_center_crop=True, do_resize=True)['pixel_values'].to(DEVICE)
            image_embeddings = model.get_image_features(inputs)
            dataset_data.append({"image_path": image_path, "image_id": image_path.split("/")[-1].split('.')[0], "embeddings": image_embeddings.squeeze()})
    
    dataset = dDataset.from_list(dataset_data)
    dataset.save_to_disk(TRAIN_DATASET_SAVE_PATH) if dataset_type == 'train' else dataset.save_to_disk(TEST_DATASET_SAVE_PATH)
    dataset = dataset.add_faiss_index("embeddings")
    return dataset

def evaluate_vlm(dataset, dialogue_data):
    # Evaluation metric is precision@30
    correct = 0
    total = 0
    model, _ = load_vlm(VLM_MODEL, DEVICE, CHECKPOINT_DIR)
    model.eval()
    with torch.no_grad():
        for dialogue in tqdm(dialogue_data, desc="Evaluating"):
            query = dialogue["image_descriptor"].split("{")[1].split("}")[0].replace("}", "").replace("\n", "").strip()
            search_results = search(dataset, query)[1]['image_id']
            for result in search_results:
                if result == dialogue["photo_id"]:
                    correct += 1
                    break
            total += 1
            if total == 50:
                break
    precision = correct / total
    print(f"Avg. Precision@30: {precision}")

def predict(dataset, dialogue_data, model=None):
    predictions, prediction_ids = [], []
    if model is None:
        model, _ = load_vlm(VLM_MODEL, DEVICE, CHECKPOINT_DIR)
    model.eval()
    with torch.no_grad():
        for dialogue in tqdm(dialogue_data):
            query = dialogue["image_descriptor"].split("{")[1].split("}")[0].replace("}", "").replace("\n", "").strip()
            search_results = search(dataset, query)[1]['image_id']
            predictions.append(" ".join(search_results))
            prediction_ids.append(dialogue["dialogue_id"])
    df = pd.DataFrame({"dialogue_id": prediction_ids, "photo_id": predictions})
    df.to_csv("predictions.csv", index=False)

if __name__ == "__main__":
    train_dialogue_data = load_dialogue_data(TRAIN_DATA_FILE)
    test_dialogue_data = load_dialogue_data(TEST_DATA_FILE)
    model = train_clip_model(train_dialogue_data)
    train_image_db = prepare_img_db(TRAIN_IMAGE_DIR, dataset_type='train')
    test_image_db = prepare_img_db(TEST_IMAGE_DIR, dataset_type='test')
    # train_image_db = Dataset.load_from_disk(TRAIN_DATASET_SAVE_PATH)
    # train_image_db = train_image_db.add_faiss_index("embeddings")
    # test_image_db = Dataset.load_from_disk(TEST_DATASET_SAVE_PATH)
    # test_image_db = test_image_db.add_faiss_index("embeddings")

    # Evaluate the model
    evaluate_vlm(train_image_db, train_dialogue_data)
    predict(test_image_db, test_dialogue_data)