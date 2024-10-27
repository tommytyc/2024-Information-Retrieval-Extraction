---
title: "IR HW1: Vector Model & BM25 Implementation"
date: 2024-10-25
author: Yu-Chien Tang
---
## IR HW1: Vector Model & BM25 Implementation

### What kind of pre-processing did you apply to the document data or question text? Additionally, please discuss how different preprocessing methods affected the performance of the models?
- I adopted simple pre-processing method to the wiki documents. Specifically, I first employed `BeautifulSoup` as a parser to remove the html tag inside the doc. Then, I removed the linebreak, tab, some redundant symbol, and the URL.
- I also noticed that the wiki docs typically included some irrelevant text like navigation bar or reference. However, I didn't find a good parsing method to remove these text and decided to leave it in future enhancement.
- The pre-processing code snippet
    ```python
    def preprocess_html(html):
        soup = BeautifulSoup(html, 'html.parser')
        txt = soup.get_text()
        txt = txt.replace('\n', ' ').replace('\t', ' ').replace('``', '"').replace("''", '"')
        # Remove URLs
        txt = ' '.join([word for word in txt.split() if not word.startswith('http')])
        return txt
    ```
- I found that without any pre-processing (only lower()) can have superior performance on the kaggle leaderboard, even better than the HTML and URL removing pre-processing method I adopted above. However, I temporarily regarded it as an overfitting and kept optimize my BM25 parameter combination.

### Please provide details on how you implemented the vector model and BM25.
> The following discussion will be more comprehensive when accompanied by the code.
#### Vector Model
- I first use `build_vocabulary` function to generate a unique list of terms across all documents and map each term to an index position. This vocabulary will define the vector space where documents and queries are represented. 
- Then I employed the `compute_tf` and `compute_idf` function for TF-IDF calculation. The `compute_tfidf` will use the TF-IDF term weight to generate the vector for each doc or query.
- During `train`, the TF-IDF vectors are precomputed for all documents in the collection and stored in `self.document_vectors`, forming the basis for similarity comparison.
- In `vector_similarity`, a query is also transformed into a TF-IDF vector. Then, cosine similarity is computed between the query vector and each document vector. This similarity score indicates how closely a document matches the query, allowing for partial matches and ranking of documents by similarity.

#### BM25
- The IDF calculation for BM25 is the same as in the TF-IDF vector model.
    - The term frequency for a document `doc_tf` is extracted from `self.document_vectors`, precomputed using the TF-IDF process.
- `bm25_score` calculates the BM25 score for a given query and document. It uses parameters `k1` and `b` to control term frequency weight and document length normalization.
- `get_top_answers` combines BM25 and vector model scores (adjustable through `self.bm25_lambda`), providing a final ranking of documents for each query. Top3-scoring documents are returned as the best answers.

#### Experiment
- To accelerate the experiment, I pre-calculate the idf weight and save it in pkl format.
- To get a better performance, I use [wandb](https://wandb.ai/) to sweep the best parameter combination (`k1`, `b`, and `bm25_lambda`). The sweep strategy is bayes optimization.

### Compare the strengths and weaknesses of the vector model and BM25. What factors might account for the differences in their performance?
> For convenience, I include the second question into the strength and weakness discussion.
#### Vector Model
- **Strength**
    - The vector model allows partial matches by scoring documents based on the cosine similarity between the query and document vectors. This can be beneficial for queries where exact term matches are not crucial.
- **Weakness**
    - TF-IDF weighting can overweight terms that appear very frequently in a document, which may reduce relevance. BM25 addresses this through saturation control parameters k1.

#### BM25
- **Strength**
    - Adjusting `k1` and `b`allows flexible control over term frequency weight saturation and document length normalization.
    - It is advantageous for queries with specific terms, especially when seeking exact matches.
- **Weakness**
    - Tuning `k1` and `b` for optimal performance is a dataset-dependent task and takes time to find the best parameter combination. In contrast, the vector model is generally robust with standard TF-IDF weights without extensive tuning.