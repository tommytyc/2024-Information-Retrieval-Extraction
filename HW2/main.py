from langchain_community.document_loaders import JSONLoader
import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from tqdm import tqdm
import os
import json
import pandas as pd
from itertools import chain
import gc
from transformers import AutoTokenizer
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from dataset import FactCheckDataset
from model import DebertaClassifier
from util import set_seed, colorprint, compute_metric, BATCH_SIZE, EPOCHS, LR, SEED, MAX_SEQ_LEN
from pytorch_metric_learning import losses
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def contrastive_loss(q, p):
    batch_size = q.size(0)
    mask = torch.eye(batch_size, dtype=torch.bool).to(device)
    q_norm = q / q.norm(dim=1)[:, None]
    p_norm = p / p.norm(dim=1)[:, None]
    sim_mat = torch.mm(q_norm, p_norm.transpose(0,1))
    nominator = (mask * torch.exp(sim_mat / 0.8)).sum(dim=1)
    denominator = (~mask * torch.exp(sim_mat / 0.8)).sum(dim=1) + 1e-6 + nominator
    all_losses = -torch.log(nominator / denominator)
    loss = torch.sum(all_losses) / batch_size
    return loss

def load_most_similar_premise(metadata_content: dict):
    claim = metadata_content['claim']
    docs_filenames = [v for k, v in metadata_content['premise_articles'].items()]
    docs = [JSONLoader(file_path=os.path.join('data/articles', filename), jq_schema='.[]').load() for filename in docs_filenames]
    docs = list(chain.from_iterable(docs))
    if not docs:
        return None
    k = min(len(docs), 3)
    embeddings = HuggingFaceEmbeddings(model_name='models--intfloat--multilingual-e5-large/snapshots/ab10c1a7f42e74530fe7ae5be82e6d4f11a719eb/', model_kwargs={'device': 'cuda'}, encode_kwargs={'normalize_embeddings': True})
    index = faiss.IndexFlatL2(len(embeddings.embed_query("hello world")))
    vectorstore = FAISS(
        embedding_function=embeddings,
        index=index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={},
    )
    vectorstore.add_documents(documents=docs, ids=[str(i) for i in range(len(docs))])
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k},
    )
    sim_premise = retriever.invoke(claim)
    del retriever
    gc.collect()
    torch.cuda.empty_cache()
    return sim_premise

def preprocess_train_valid_test_data(data_dir: str):
    train_json, valid_json, test_json = [], [], []
    
    print("### Processing train_data.json")
    with open(os.path.join(data_dir, 'train.json'), 'r') as f:
        train_json_file = json.load(f)
    for d in tqdm(train_json_file):
        sim_premise = load_most_similar_premise(d['metadata'])
        if not sim_premise:
            continue
        train_json.append({'claim_id': d['label']['id'], 'claim': d['metadata']['claim'], 'top3_premise': [d.page_content for d in sim_premise], 'label': d['label']['rating']})
    with open(os.path.join(data_dir, 'train_data.json'), 'w') as f:
        json.dump(train_json, f)
    
    # print("### Processing valid_data.json")
    # with open(os.path.join(data_dir, 'valid.json'), 'r') as f:
    #     valid_json_file = json.load(f)
    # for d in tqdm(valid_json_file):
    #     sim_premise = load_most_similar_premise(d['metadata'])
    #     if not sim_premise:
    #         continue
    #     valid_json.append({'claim_id': d['metadata']['id'], 'claim': d['metadata']['claim'], 'top3_premise': [d.page_content for d in sim_premise], 'label': d['label']['rating']})
    # with open(os.path.join(data_dir, 'valid_data.json'), 'w') as f:
    #     json.dump(valid_json, f)
    
    # print("### Processing test_data.json")
    # with open(os.path.join(data_dir, 'test.json'), 'r') as f:
    #     test_json_file = json.load(f)
    # for d in tqdm(test_json_file):
    #     sim_premise = load_most_similar_premise(d['metadata'])
    #     if not sim_premise:
    #         continue
    #     test_json.append({'claim_id': d['metadata']['id'], 'claim': d['metadata']['claim'], 'top3_premise': [d.page_content for d in sim_premise]})
    # with open(os.path.join(data_dir, 'test_data.json'), 'w') as f:
    #     json.dump(test_json, f)

def prepare_train_valid_data(train_path, valid_path, batch_size):
    train_data = FactCheckDataset(train_path, mode='train')
    dev_data = FactCheckDataset(valid_path, mode='valid')
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    valid_dataloader = DataLoader(dev_data, batch_size=1, shuffle=False)
    return train_dataloader, valid_dataloader

def prepare_test_data(test_path):
    test_data = FactCheckDataset(test_path, mode='test')
    test_dataloader = DataLoader(test_data, batch_size=1, shuffle=False)
    return test_dataloader

def train(model, train_dataloader, valid_dataloader):
    tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=LR)
    cls_loss_fn = nn.CrossEntropyLoss()
    SupConLoss = losses.SupConLoss(temperature=0.1)
    best_macro_f1 = 0.0
    with tqdm(range(EPOCHS), desc='Epoch') as tepoch:
        for epoch in tepoch:
            model.train()
            total_loss = 0
            for data in tqdm(train_dataloader):
                optimizer.zero_grad()
                premise, claim = list(map(list, zip(*data['premise']))), data['claim']
                input_text = ["[CLS] " + "\n".join(p) + " [SEP] " + c + " [SEP]" for p, c in zip(premise, claim)]
                input_text = tokenizer(input_text, padding=True, truncation=True, max_length=MAX_SEQ_LEN, return_tensors='pt').to(device)
                label = F.one_hot(torch.as_tensor(data['label']), num_classes=3).to(device)
                logits, hidden_state = model(input_text)
                q1 = F.dropout(hidden_state, p=0.1, training=True)
                q2 = F.dropout(hidden_state, p=0.1, training=True)
                # loss = cls_loss_fn(F.softmax(logits, dim=1), label.float()) + 0.5 * SupConLoss(hidden_state, torch.as_tensor(data['label']).float())
                loss = cls_loss_fn(F.softmax(logits, dim=1), torch.as_tensor(data['label']).to(device)) + 0.5 * contrastive_loss(q1, q2)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            model.eval()
            labels, preds = [], []
            with torch.no_grad():
                for data in valid_dataloader:
                    premise, claim = list(map(list, zip(*data['premise']))), data['claim']
                    input_text = ["[CLS] " + "\n".join(p) + " [SEP] " + c + " [SEP]" for p, c in zip(premise, claim)]
                    input_text = tokenizer(input_text, padding=True, truncation=True, max_length=MAX_SEQ_LEN, return_tensors='pt').to(device)
                    label = data['label']
                    logits, _ = model(input_text)
                    pred = torch.argmax(F.softmax(logits.cpu(), dim=1), 1)
                    labels.append(label)
                    preds.append(pred.detach().cpu().numpy())
            macro_f1 = compute_metric(labels, preds)        
            tepoch.set_postfix(loss=total_loss / len(train_dataloader), macro_f1=macro_f1)
            if macro_f1 > best_macro_f1:
                best_macro_f1 = macro_f1
                torch.save(model.state_dict(), f'model/model_{best_macro_f1}.pth')
    return model

def test(model_path, test_dataloader):
    model = DebertaClassifier()
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")
    model.eval()
    cids, labels, preds = [], [], []
    with torch.no_grad():
        for data in tqdm(test_dataloader):
            cid, premise, claim = data['claim_id'], list(map(list, zip(*data['premise']))), data['claim']
            input_text = ["[CLS] " + "\n".join(p) + " [SEP] " + c + " [SEP]" for p, c in zip(premise, claim)]
            input_text = tokenizer(input_text, padding=True, truncation=True, max_length=MAX_SEQ_LEN, return_tensors='pt').to(device)
            logits, _ = model(input_text)
            pred = torch.argmax(F.softmax(logits.cpu(), dim=1), 1)
            preds.append(pred.detach().cpu().numpy().item())
            cids.append(cid.item())
        df = pd.DataFrame({'id': cids, 'rating': preds})
        df.to_csv('submission.csv', index=False)

if __name__ == "__main__":
    # preprocess_train_valid_test_data('data/')
    set_seed(SEED)
    train_loader, dev_loader = prepare_train_valid_data('data/train_data.json', 'data/valid_data.json', BATCH_SIZE)
    model = DebertaClassifier()
    model = train(model, train_loader, dev_loader)
    # test_loader = prepare_test_data('data/test_data.json')
    # test('model/model_0.5519.pth', test_loader)