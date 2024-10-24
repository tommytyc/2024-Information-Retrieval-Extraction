import numpy as np
import cupy as cp
import pandas as pd
import os
from tqdm import tqdm
import pickle as pkl
from collections import Counter
import wandb

class QASystem:
    def __init__(self, filename, k1=1.75, b=0.1):
        self.vocabulary = {}
        self.idf = None
        self.document_vectors = None
        self.documents = None
        self.k1 = k1
        self.b = b
        self.avgdl = None
        self.documents = self.load_documents(filename)
        self.train()

    def load_documents(self, filename):
        documents = []
        df = pd.read_csv(filename)
        for _, row in df.iterrows():
            documents.append(row['Document_HTML'])  # Assuming the CSV has a 'content' column
        return documents

    def preprocess(self, text):
        return text.lower().split()

    def build_vocabulary(self, documents):
        word_set = set()
        for doc in documents:
            word_set.update(doc)
        self.vocabulary = {word: idx for idx, word in enumerate(word_set)}

    def compute_tf(self, document):
        word_counts = Counter(document)
        return {word: count / len(document) for word, count in word_counts.items()}

    def compute_idf(self, documents):
        N = len(documents)
        idf = {}
        for word in tqdm(self.vocabulary):
            doc_count = sum(1 for doc in documents if word in doc)
            idf[word] = np.log((N - doc_count + 0.5) / (doc_count + 0.5) + 1)
        return idf

    def compute_tfidf(self, tf, idf):
        tfidf = np.zeros(len(self.vocabulary))
        for word, tf_value in tf.items():
            if word in self.vocabulary:
                idx = self.vocabulary[word]
                tfidf[idx] = tf_value * idf[word]
        return tfidf

    def train(self):
        self.documents = [self.preprocess(doc) for doc in self.documents]
        self.build_vocabulary(self.documents)
        if os.path.exists("/mnt/NAS/yctang/work/IR/HW1/idf.pkl"):
            with open("/mnt/NAS/yctang/work/IR/HW1/idf.pkl", "rb") as f:
                self.idf = pkl.load(f)
                for k, v in tqdm(self.idf.items()):
                    self.idf[k] = cp.asnumpy(v)
        else:
            self.idf = self.compute_idf(self.documents)
            with open("/mnt/NAS/yctang/work/IR/HW1/idf.pkl", "wb") as f:
                pkl.dump(self.idf, f)
        self.document_vectors = np.array([self.compute_tfidf(self.compute_tf(doc), self.idf) for doc in tqdm(self.documents)])
        self.avgdl = np.mean([len(doc) for doc in self.documents])

    def bm25_score(self, query, doc_idx):
        query_tf = self.compute_tf(query)
        doc = self.documents[doc_idx]
        doc_len = len(doc)
        scores = []
        for term, tf in query_tf.items():
            if term not in self.vocabulary:
                continue
            term_idx = self.vocabulary[term]
            doc_tf = self.document_vectors[doc_idx, term_idx]
            idf = self.idf[term]
            numerator = idf * doc_tf * (self.k1 + 1)
            denominator = doc_tf + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)
            scores.append(numerator / denominator)
        return np.sum(scores)

    def vector_similarity(self, query):
        query_vector = self.compute_tfidf(self.compute_tf(query), self.idf)
        return np.dot(self.document_vectors, query_vector) / (np.linalg.norm(self.document_vectors, axis=1) * np.linalg.norm(query_vector))

    def get_top_answers(self, queries, top_k=3):
        queries = [self.preprocess(q) for q in queries]
        bm25_scores = [np.array([self.bm25_score(q, i) for i in range(len(self.documents))]) for q in queries]
        vector_scores = np.array([self.vector_similarity(q) for q in queries])
        combined_scores = 0.5 * np.reshape(bm25_scores, (len(queries), -1)) + 0.5 * np.reshape(vector_scores, (len(queries), -1))
        top_indices = [np.argsort(combined_scores[k])[-top_k:][::-1] for k in range(len(queries))]
        return [[(i+1, self.documents[i], combined_scores[k][i]) for i in top_indices[k]] for k in range(len(queries))]
    
    def test(self, filename, sweep=False):
        df = pd.read_csv(filename)
        queries = df['Question'].tolist()
        results = self.get_top_answers(queries)
        predictions = []
        for r in results:
            pred = []
            for i, _, _ in r:
                pred.append(str(i))
            predictions.append(" ".join(pred))
        if not sweep:
            df_pred = pd.DataFrame({"answer": predictions})
            df_pred.index += 1
            df_pred.index.name = "index"
            df_pred.to_csv("predictions.csv", index=True)
        else:
            answer = df['Answer ID'].tolist()
            score = self.evaluate(predictions, answer)
            wandb.log({"recall@3": score})

    def evaluate(self, predictions, answers):
        # evaluation metric is recall@3
        correct = 0
        for pred, ans in zip(predictions, answers):
            p = pred.split()
            if str(ans) in p:
                correct += 1
        return correct / len(answers)
    
def sweep_main():
    wandb.init(project="IR_HW1")
    qa_system = QASystem(filename=("/mnt/NAS/yctang/work/IR/HW1/data/documents_data.csv"), k1=wandb.config.k1, b=wandb.config.b)
    qa_system.test(filename="/mnt/NAS/yctang/work/IR/HW1/data/train_question.csv", sweep=True)

def sweep_wandb():
    wandb.login()
    sweep_config = {
        "name": "IR_HW1",
        "method": "bayes",
        "metric": {"name": "recall@3", "goal": "maximize"},
        "parameters": {
            "k1": {"min": 0.00, "max": 3.00},
            "b": {"min": 0.00, "max": 1.00}
        }
    }
    sweep_id = wandb.sweep(sweep_config, project="IR_HW1")
    wandb.agent(sweep_id, function=sweep_main)


if __name__ == "__main__":
    # sweep_wandb()
    qa_system = QASystem(filename=("/mnt/NAS/yctang/work/IR/HW1/data/documents_data.csv"), k1=0.0032726, b=0.37489)
    qa_system.test(filename="/mnt/NAS/yctang/work/IR/HW1/data/test_question.csv")
    
    # test_query = ["which is the most common use of opt-in e-mail marketing", "how i.met your mother who is the mother"]
    # results = qa_system.get_top_answers(test_query)
    # for r in results:
    #     for i, doc, score in r:
    #         print(f"Score: {score:.4f}, Answer: {' '.join(doc)[:40]}")