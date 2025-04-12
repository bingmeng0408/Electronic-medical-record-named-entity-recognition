#  Electronic Medical Record Named Entity Recognition

​     本项目主要实现使用了基于Bert与CRF模型的网络，利用viterbi算法进行的命名实体识别项目，以推进电子病历结构化的发展。该项目提供了原始训练数据样本(一般醒目,出院情况,病史情况,病史特点,诊疗经过)与转换版本,训练脚本,预训练模型,可用于命名实体识别的研究.

## 项目介绍

​      电子病历结构化是让计算机理解病历、应用病历的基础。基于对病历的结构化，可以计算出症状、疾病、药品、检查检验等多个知识点之间的关系及其概率，构建医疗领域的知识图谱，进一步优化医生的工作. CCKS2018的电子病历命名实体识别的评测任务，是对于给定的一组电子病历纯文本文档，识别并抽取出其中与医学临床相关的实体，并将它们归类到预先定义好的类别中。数据集来自Medical Named Entity Recognition implement using bi-directional lstm and crf model with char embedding.CCKS2017中文电子病例命名实体识别项目，共提供了600份标注好的电子病历文本，共需识别含解剖部位、独立症状、症状描述、手术和药物五类实体。 领域命名实体识别问题自然语言处理中经典的序列标注问题, 本项目是运用深度学习方法进行命名实体识别的一个尝试.

## 实验数据

### 一、目标序列标记集合 

​       O非实体部分,TREATMENT治疗方式, BODY身体部位, SIGN疾病症状, CHECK医学检查, DISEASE疾病实体。

### 二、序列标记方法 

​       采用BIO三元标记

## 环境配置

```python
torh  2.5.1 
python 3.11
transformers 4.27.4
torchcrf 1.1.0 
scikit-learn 1.2.2
```

## 文件介绍

> bert-base-chinese：Bert预训练模型，可自行从https://huggingface.co/下载biobert、medbert、roberta等模型替换，用于性能提升
>
> data：转换后，用于训练的数据集
>
> data_origin：原始数据集
>
> save_model：保存训练模型
>
> Bert_module.py：搭建Bert_crf模型
>
> config.py：配置文件
>
> predict_sample.py：预测文件
>
> train.py：训练文件
>
> transfer_data.py：数据转换文件

## 测试结果

测试集f1 score为97%

## 预测效果

为显示准确性，以下选取了数据集中的一个句子，进行预测，模型效果欢迎大家在predict_sample.py中随机输入句子进行验证。

> ```
> sample="患者精神状况好，无发热，诉右髋部疼痛，饮食差，二便正常，查体：神清，各项生命体征平稳，心肺腹查体未见异常。右髋部压痛，右下肢皮牵引固定好，无松动，右足背动脉搏动好，足趾感觉运动正常。")
> label
> 发热  9  10 症状和体征
> 右髋部 13 15 身体部位
> 疼痛  16 17 症状和体征
> 二便  23 24 身体部位
> 查体  28 29 检查和检验
> 心肺腹查体   43 47 检查和检验
> 右髋部 53 55 身体部位
> 压痛  56 57 检查和检验
> 右下肢 59 61 身体部位
> 右足背动脉   73 77 身体部位
> 足趾  82 83 身体部位
> 预测结果：
> [['发热', '症状和体征'], ['右髋部', '身体部位'], ['疼痛', '症状和体征'], ['二便', '身体部位'], ['查体', '检查和检验'], ['心肺腹查体', '检查和检验'], ['右髋部', '身体部位'], ['压痛', '检查和检验'], ['右下肢', '身体部位'], ['右足背动脉', '身体部位'], ['足趾', '身体部位']]
> ```

