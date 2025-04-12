import torch
import torch.nn as nn
from TorchCRF import CRF
from config import *
from Bert_ner.dataloader import *
from transformers import BertModel
conf = Config()
class BERT_CRF(nn.Module):
    def __init__(self, dropout, tag2id):
        super(BERT_CRF, self).__init__()
        self.tag_to_ix = tag2id
        self.tag_size = len(tag2id)
        self.bert = BertModel.from_pretrained(conf.bert_path)
        self.bert_dim = 768
        self.dropout = nn.Dropout(dropout)
        self.hidden2tag = nn.Linear(self.bert_dim, self.tag_size)
        self.crf = CRF(self.tag_size)

    def forward(self, x, mask):
        bert_output = self.bert(x, attention_mask=mask)
        bert_embeddings = bert_output.last_hidden_state
        outputs = self.dropout(bert_embeddings)
        outputs = self.hidden2tag(outputs)
        outputs = outputs * mask.unsqueeze(-1)
        outputs = self.crf.viterbi_decode(outputs, mask)
        return outputs
    def log_likelihood(self, x, tags, mask):
        bert_output = self.bert(x, attention_mask=mask)
        bert_embeddings = bert_output.last_hidden_state
        outputs = self.dropout(bert_embeddings)
        outputs = self.hidden2tag(outputs)
        outputs = outputs * mask.unsqueeze(-1)
        return -self.crf(outputs, tags, mask)


if __name__ == '__main__':
    conf = Config()
    model = BERT_CRF(
        dropout=conf.dropout,
        tag2id=conf.tag2id
    )

    train_dataloader, dev_dataloader = get_data()
    for x, y, mask in train_dataloader:
        mask = mask.to(torch.bool)
        loss = model.log_likelihood(x, y, mask)
        print(loss)
        break