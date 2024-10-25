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

### Please provide details on how you implemented the vector model and BM25.
> The following discussion will be more comprehensive when accompanied by the code.
#### Vector Model
- I first use `build_vocabulary` function to generate a unique list of terms across all documents and map each term to an index position. This vocabulary will define the vector space where documents and queries are represented. 
- Then I employed the `compute_tf` and `compute_idf` function for tf-idf calculation. The `compute_tfidf` will use the tf-idf term weight to generate the vector for each doc or query.
- During `train`, the tf-idf vectors are precomputed for all documents in the collection and stored in `self.document_vectors`, forming the basis for similarity comparison.
- In `vector_similarity`, a query is also transformed into a tf-idf vector. Then, cosine similarity is computed between the query vector and each document vector. This similarity score indicates how closely a document matches the query, allowing for partial matches and ranking of documents by similarity.

#### BM25
- 

### Compare the strengths and weaknesses of the vector model and BM25. What factors might account for the differences in their performance?
- pass