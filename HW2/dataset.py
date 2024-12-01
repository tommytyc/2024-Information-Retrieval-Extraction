from torch.utils.data import Dataset, DataLoader
import json

class FactCheckDataset(Dataset):
    def __init__(self, file_path, mode='train'):
        super().__init__()
        with open(file_path, 'r') as f:
            data = json.load(f)
        self.mode = mode
        self.data = []
        for d in data:
            if len(d['top3_premise']) != 3 and mode == 'train':
                continue
            self.data.append(d)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        if self.mode != 'test':
            return {
                'claim_id': self.data[idx]['claim_id'],
                'premise': self.data[idx]['top3_premise'],
                'claim': self.data[idx]['claim'],
                'label': self.data[idx]['label']
            }
        else:
            return {
                'claim_id': self.data[idx]['claim_id'],
                'premise': self.data[idx]['top3_premise'],
                'claim': self.data[idx]['claim']
            }

if __name__ == '__main__':
    dataset = FactCheckDataset('data/valid_data.json', mode='valid')
    train_dataloader = DataLoader(dataset, batch_size=4, shuffle=False)
    for i, data in enumerate(train_dataloader):
        # print(data['premise'])
        print(list(map(list, zip(*data['premise']))))
        break