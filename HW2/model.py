from transformers import AutoModel
import torch.nn as nn

class DebertaClassifier(nn.Module):
    def __init__(self):
        super(DebertaClassifier, self).__init__()
        self.transformer = AutoModel.from_pretrained('microsoft/deberta-v3-base')
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(768, 3)
    
    def forward(self, input_text):
        hidden_states = self.transformer(**input_text)[0][:, 0]
        # hidden_states = self.dropout(hidden_states)
        logits = self.classifier(hidden_states)
        return logits, hidden_states