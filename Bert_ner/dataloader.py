import random
from Bert_ner.config import *
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer
conf = Config()
tokenizer = BertTokenizer.from_pretrained(conf.bert_path)
def build_data():
    datas = []
    sample_x = []
    sample_y = []
    for line in open(conf.train_path, 'r', encoding='utf-8'):
        line= line.rstrip().split('\t')
        if not line:
            continue
        char = line[0]
        if not char:
            continue
        cate = line[-1]
        sample_x.append(char)
        sample_y.append(cate)
        if char in ['。', '?', '!', '！', '？']:
            datas.append([sample_x, sample_y])
            sample_x = []
            sample_y = []
    return datas

datas=build_data()
def split_data(datas):
    random.seed(42)
    total_length = len(datas)
    first_split = int(total_length * 0.8)
    random.shuffle(datas)
    train_data = datas[:first_split]
    dev_data = datas[first_split:]
    return train_data, dev_data
train_data, dev_data= split_data(datas)
class NerDataset(Dataset):
    def __init__(self, datas):
        super().__init__()
        self.datas = datas

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, item):
        text = self.datas[item][0]
        labels = self.datas[item][1]
        return text, labels


def collate_fn(batch):
    texts = [item[0] for item in batch]
    labels_list = [item[1] for item in batch]
    data_dict = tokenizer.batch_encode_plus(
        texts,
        is_split_into_words=True,
        add_special_tokens=False,
        return_attention_mask=True,
        return_tensors='pt',
        padding=True,
        truncation=False,
    )
    input_ids = data_dict['input_ids']
    attention_mask = data_dict['attention_mask']
    padded_labels = []
    for labels, input_id in zip(labels_list, input_ids):
        label_ids = [conf.tag2id[label] for label in labels]
        aligned_labels = label_ids+(len(input_id)-len(label_ids))*[0]
        padded_labels.append(aligned_labels)
    padded_labels = torch.tensor(padded_labels, dtype=torch.long)
    return input_ids, padded_labels, attention_mask
def get_data():
    train_dataset = NerDataset(train_data)
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=conf.batch_size,
        collate_fn=collate_fn,
        drop_last=False,
    )

    dev_dataset = NerDataset(dev_data)
    dev_dataloader = DataLoader(
        dataset=dev_dataset,
        batch_size=conf.batch_size,
        collate_fn=collate_fn,
        drop_last=False,
    )
    return train_dataloader, dev_dataloader
if __name__ == '__main__':
    train_dataloader, dev_dataloader= get_data()
    for input_ids, labels, attention_mask in train_dataloader:
        print(input_ids.shape)
        print(labels.shape)
        print(attention_mask.shape)

