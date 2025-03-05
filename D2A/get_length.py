import os
import get_dataset as gds
import torch
from transformers import AutoModel, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
Codebert_model = AutoModel.from_pretrained("microsoft/codebert-base")
Codebert_model.cuda(0)
sentenselist,labellist,filenamelist = gds.read_dataset()
length_list = []
all_list=[]
for sentense,label,filename in zip(sentenselist,labellist,filenamelist):
    
    shape=(1,0,768)
    embedding=torch.rand(shape).cuda(0)
    for a_sentense in sentense:
        nl_tokens=tokenizer.tokenize(a_sentense)
        tokens=[tokenizer.cls_token]+nl_tokens+[tokenizer.sep_token]
        tokens_ids=tokenizer.convert_tokens_to_ids(tokens)
        context_embeddings=Codebert_model(torch.tensor(tokens_ids).cuda(0)[None,:])[0].cuda(0)
        embedding = torch.cat([embedding,context_embeddings],dim=1)
    length= embedding.size()[1]
    print(length)
    length_list.append(length)
    all_list.append([filename,label,length])

length_list.sort()
s=set(length_list)
for i in s:
    print(i,"is",length_list.count(i))
all_list=sorted(all_list,key=lambda x:(-x[2],x[1]))
f=open("/home/ao_ding/gwork/data/D2A/data_a/all_length.txt","w")
for line in all_list:
    f.write(str(line)+'\n')
f.close()
