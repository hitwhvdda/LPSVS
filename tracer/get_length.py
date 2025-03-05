import os
import get_dataset as gds
import torch
from transformers import AutoModel, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
Codebert_model = AutoModel.from_pretrained("microsoft/codebert-base")
Codebert_model.cuda(0)
dataset = gds.all_dataset(0.8,0.1,0.1,True,True)
length_list = []
for data in dataset:
    sentense,label,_=data
    shape=(1,0,768)
    embedding=torch.rand(shape).cuda(0)
    for a_sentense in sentense:
        nl_tokens=tokenizer.tokenize(a_sentense)
        tokens=[tokenizer.cls_token]+nl_tokens+[tokenizer.sep_token]
        tokens_ids=tokenizer.convert_tokens_to_ids(tokens)
        context_embeddings=Codebert_model(torch.tensor(tokens_ids).cuda(0)[None,:])[0].cuda(0)
        embedding = torch.cat([embedding,context_embeddings],dim=1)
    len = embedding.size()[1]
    print(len)
    length_list.append(len)
length_list.sort()
print(length_list[-1])
