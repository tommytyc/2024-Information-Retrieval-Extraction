---
title: "IR HW2: Fact Checking"
date: 2024-12-01
author: Yu-Chien Tang
---
## IR HW2: Fact Checking

### Describe your approach to data preprocessing and information retrieval. Please choose at least 2 of any IR methods and compare their performance.
#### Embedding Similarity Retrieval
- Inspired by vanilla RAG approach, I employed an embedding model to retrieve the most relevant premise sentences for each claim. 
    - Specifically, each claim sentence and its corresponding premise sentences would be fed to a transformer-based encoder (`multilingual-e5-large` here) to get the sentence embeddings. Then, the similarity between the claim embedding and the premise sentence embeddings would be calculated via a similarity function (`cosine similarity` here). The top-3 similar premise sentences would be extracted as part of the training data in our downstream task setting.
    - I adopted [langchain](https://www.langchain.com/) as the retrieval implementation framework. As for the vector store, I used [FAISS](https://github.com/facebookresearch/faiss) to accelerate the similarity calculation.
    - I observed that some of the claim had less than 3 premise sentences, and hence decided to set the top-`k` similarity retrieval as top-`min(premise_counts, 3)`.

#### BM25
- Although I had tried to used BM25 as my implementation in HW1, I immediately found that the inverse document frequency could take way too long time to calculate. Therefore I chose the Embedding Similarity Retrieval as my main method.

### Describe your approach to claim prediction. Details such as model selection, hyperparameters should be provided.
- I formulated the claim prediction task as a classic text classification task, and decided to adopt a transformer encoder to encode the claim and top-3 premise then classify it.
    - Concretely, the input would first be transformed into `input_text = ["[CLS] " + "\n".join(p) + " [SEP] " + c + " [SEP]" for p, c in zip(premise, claim)]`
    - Then, I used `deberta-v3-base` as the encoder to encode the `input_text` into a 768-dimension vector. A linear layer and a softmax layer would then classify the vector into three category, and the category with largest probability would become the final prediction.
    - I chose cross entropy loss as the main optimization objective function. Besides, inspired by [SimCSE](https://aclanthology.org/2021.emnlp-main.552/), I employed a dropout function twice to get different augmentation view of the `input_text` vector. Then, the two augmentation view would be optimized by a InfoNCE loss to make them as similar as possible. This could effectively improve the training robustness.
    ```python
    logits, hidden_state = model(input_text)
    q1 = F.dropout(hidden_state, p=0.1, training=True)
    q2 = F.dropout(hidden_state, p=0.1, training=True)
    loss = cls_loss_fn(F.softmax(logits, dim=1), torch.as_tensor(data['label']).to(device)) + 0.5 * contrastive_loss(q1, q2)
    ```

### Do error analysis or case study. Is there anything worth mentioning while checking the mispredicted data? Share with us. Anytime you try to make a conclusion about the data or model, you should provide concrete data example.
#### Case: claim_id `29071`
> St Austin University North Carolina says eating vaginal fluid makes you immune to cancer
- Top-3 similar premise
    - . Eating vaginal fluids makes you immune to cancer, and other
    - Eating vaginal fluids makes you immune to cancer, and other diseases.
    - Eating vaginal fluids makes you immune to cancer, and other diseases.
- Results
    - Prediction: `Partial True`
    - Label:      `False`  
- Analysis
    - The retrieval results show that the embedding model can effectively find premise sentences relevant to the claim. However, the top-3 retrieval results do not support the fact about `St Austin University` and only show sentences relevant to the claim that `Eating vaginal fluids can immune to cancer`.
    - I hypothesize that the model learns to predict `Partial True` when it can only detect partial facts, as demonstrated in the example above. However, to successfully predict the actual label `False`, we need to include more premise sentences in this example. This indicates a future exploration direction: either including more premise sentences during training or leveraging LLMs to retrieve other sentences that may not appear similar to the claim but can actually help the model discern the facts.