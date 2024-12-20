---
title: "IR HW3: Dialogue-Based Photo Retrieval"
date: 2024-12-20
author: Yu-Chien Tang
---
## IR HW3: Dialogue-Based Photo Retrieval

### What kind of pre-processing did you apply to the photo or dialogue text? Additionally, please discuss how different preprocessing methods affected the performance of the models?
#### VisualDialog
- This task aims to retrieve the most relevant photo to share based on the dialogue history. Inspired by [VisualDialog](https://aclanthology.org/2024.findings-acl.700/), I leverage the robust reasoning ability of LLM to generate precise dialogue-associated visual descriptors, facilitating seamless connection with image.
- Specifically, the LLM (I use `Meta-Llama-3.3-70B-Instruct` via [Sambanova API](https://api.sambanova.ai/v1)) will be instructed to zero-shot generate the visual descriptors in five dimensions: `main subject`, `prominent objects in the foreground`, `background scene`, `events`, and `materials and attributes`. The generated visual descriptors will serve as a cue to guide the image retrieval. 
- The visual descriptors generation prompt:
    ```python
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
    ```
#### W/O preprocessing
- If the dialogue is not properly preprocessing and directly feed it to downstream vision language model (VLM), the dialogue may contain too many noise and confuse the VLM.
- For example, if I use [CLIP](https://proceedings.mlr.press/v139/radford21a) to encode the photo and dialogue, since the max_length of CLIP tokenizer is 77, those tokens longer than 77 will be truncated, and the remaining dialogue may lose important text features that can help to retrieve the correct photo.

### How did you align the photo and dialogue text in the same embedding space? Use pretrained model or train your own?
- I employ CLIP model to deal with the multimodal features. CLIP is a dual encoder architecture composed of a vision encoder and a text encoder. In pretraining stage, it use contrastive learning to align the text embedding of image caption and vision embedding of each image. 
- I experiment with two siblings of CLIP model, including `openai/clip-vit-base-patch32` and `openai/clip-vit-large-patch14`. During finetuning, I utilize a contrastive loss to align the image embedding with the LLM-generated visual descriptor text embedding.
- The pytorch contrastive loss:
    ```python
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
    ```
- The finetuned CLIP model is saved for prediction. During inference time, the testing images will be encoded by the finetuned model to get the image embeddings and be indexed by [FAISS](https://github.com/facebookresearch/faiss) to improve the retrieval efficiency.
    - The testing dialogue visual descriptors will also be encoded by the finetuned model to get the text embedding.
    - The top-30 images which highest embedding similarity with the visual descriptor will be retrieved as the prediction.

### Please discuss based on your experimental results. How do you improve the performance of your model? (e.g. add a module or try different models and observing performance changes). What was the result?
#### Zero-shot
- I have tried to use `openai/clip-vit-large-patch14` as the base CLIP model and submit a zero-shot image retrieval results to the leaderboard. Only achieved 0.33 Precision@30.
- The results indicated that without finetuning, even a large CLIP model cannot perform well in this dataset.

#### Finetuning
- I finetuned two different size of CLIP model.
    - Batch size: 128
    - Learning rate: 5e-5
    - Epoch: 15

    | model | Precision@30 |
    | ----- | ------------ |
    | `openai/clip-vit-base-patch32` | 0.6467 |
    | `openai/clip-vit-large-patch14` | 0.7133 |

- The experimental results showcased the superiority of large CLIP model after finetuning.