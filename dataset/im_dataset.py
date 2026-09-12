import json

from torch.utils.data import Dataset
import torch
import os

#设置tokenizers并行为false，避免warning
os.environ['TOKENIZERS_PARALLELISM'] = 'false'


#先写dataset类

class PretrainDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=512):
        super().__init__()
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.max_len = max_length
        self.samples = self.load_data(data_path)
        
#实现dataset内定的方法
    def load_data(self, path):
        samples = []
        with open(path, 'r', encoding='utf-8') as f:
            for line_num,line in enumerate(f,1):
                data = json.loads(line.strip())
                samples.append(data)
        return samples
#_len_
    def __len__(self):
        return len(self.samples)
#_getitem_
    def __getitem__(self, index):
        sample = self.samples[index]

        encoding = self.tokenizer(
            str(sample['text']),
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
#(max_length,)
        input_ids = encoding['input_ids'].squeeze()

#[1,1,1,0,0]
        loss_mask = input_ids != self.tokenizer.pad_token_id

        #自回归
        X=torch.tensor(input_ids[:-1],dtype=torch.long)
        Y=torch.tensor(input_ids[1:],dtype=torch.long)

        loss_mask=torch.tensor(loss_mask[1:],dtype=torch.long)
        
        return X,Y,loss_mask