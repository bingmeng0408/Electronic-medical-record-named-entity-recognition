import torch
import torch.nn as nn
import torch.optim as optim
from Bert_ner.Bert_module import *
from Bert_ner.dataloader import *
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
from Bert_ner.config import *
import time


conf = Config()
def model2train():
    train_dataloader, dev_dataloader= get_data()
    model = BERT_CRF(conf.dropout, conf.tag2id)
    model = model.to(conf.device)
    optimizer = optim.Adam(model.parameters(), lr=conf.lr)
    start_time = time.time()
    f1_score = -1000
    for epoch in range(conf.epochs):
        model.train()
        for index, (inputs, labels, mask) in enumerate(tqdm(train_dataloader, desc='bert+crf训练')):
            x = inputs.to(conf.device)
            mask = mask.to(torch.bool).to(conf.device)
            tags = labels.to(conf.device)
            loss = model.log_likelihood(x, tags, mask).mean()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=10)
            optimizer.step()
            # if index % 250 == 0:
            #     print('epoch:%04d,------------loss:%f' % (epoch, loss.item()))
        precision, recall, f1, report = model2dev(dev_dataloader, model)
        if f1 > f1_score:
            f1_score = f1
            torch.save(model.state_dict(), conf.save_path)
            print(report)
    end_time = time.time()
    print(f'训练总耗时：{end_time - start_time}')


def model2dev(dev_iter, model):
    aver_loss = 0
    preds, golds = [], []
    model.eval()
    for index, (inputs, labels, mask) in enumerate(tqdm(dev_iter, desc="测试集验证")):
        val_x = inputs.to(conf.device)
        mask = mask.to(conf.device)
        val_y = labels.to(conf.device)
        predict = []
        mask = mask.to(torch.bool)
        predict = model(val_x, mask)
        loss = model.log_likelihood(val_x, val_y, mask)
        aver_loss += loss.mean().item()
        leng = []
        for i in val_x.cpu():
            tmp = []
            for j in i:
                if j.item() > 0:
                    tmp.append(j.item())
            leng.append(tmp)
        for index, i in enumerate(predict):
            preds.extend(i[:len(leng[index])])
        for index, i in enumerate(val_y.tolist()):
            golds.extend(i[:len(leng[index])])
    precision = precision_score(golds, preds, average='macro')
    recall = recall_score(golds, preds, average='macro')
    f1 = f1_score(golds, preds, average='macro')
    report = classification_report(golds, preds)
    return precision, recall, f1, report

if __name__ == '__main__':
    model2train()
