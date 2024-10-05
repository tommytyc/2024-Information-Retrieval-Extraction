import numpy as np
from collections import Counter

class QASystem:
    def __init__(self):
        self.vocabulary = {}
        self.idf = None
        self.document_vectors = None
        self.documents = None
        self.k1 = 1.5
        self.b = 0.75
        self.avgdl = None

    def preprocess(self, text):
        return text.lower().split()

    def build_vocabulary(self, documents):
        word_set = set()
        for doc in documents:
            word_set.update(doc)
        self.vocabulary = {word: idx for idx, word in enumerate(sorted(word_set))}

    def compute_tf(self, document):
        word_counts = Counter(document)
        return {word: count / len(document) for word, count in word_counts.items()}

    def compute_idf(self, documents):
        N = len(documents)
        idf = {}
        for word in self.vocabulary:
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

    def train(self, questions, answers):
        self.documents = [self.preprocess(q + " " + a) for q, a in zip(questions, answers)]
        self.build_vocabulary(self.documents)
        self.idf = self.compute_idf(self.documents)
        self.document_vectors = np.array([self.compute_tfidf(self.compute_tf(doc), self.idf) for doc in self.documents])
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

    def get_top_answers(self, query, top_k=5):
        query = self.preprocess(query)
        bm25_scores = np.array([self.bm25_score(query, i) for i in range(len(self.documents))])
        vector_scores = self.vector_similarity(query)
        combined_scores = 0.5 * bm25_scores + 0.5 * vector_scores
        top_indices = np.argsort(combined_scores)[-top_k:][::-1]
        return [(self.documents[i], combined_scores[i]) for i in top_indices]

# 使用示例
questions = ["What is Python?", "How do I install NumPy?", "What is machine learning?"]
answers = ["Python is a programming language.", "You can install NumPy using pip.", "Machine learning is a subset of AI."]

qa_system = QASystem()
qa_system.train(questions, answers)

test_query = "How to use Python?"
results = qa_system.get_top_answers(test_query)
for doc, score in results:
    print(f"Score: {score:.4f}, Answer: {' '.join(doc)}")