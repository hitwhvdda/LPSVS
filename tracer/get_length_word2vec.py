import os
import get_dataset as gds
import torch
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
import numpy as np


dataset = gds.all_dataset(0.8,0.1,0.1,True,True)
length_list = []
def tokenize(sentences):
    for line in sentences:
        yield word_tokenize(line.strip())

def sentence_embedding(model, sentence):
    return np.mean([model.wv[word] for word in sentence if word in model.wv], axis=0)
for data in dataset:
    sentense,label,_=data
    shape=(1,0,300)
    embedding=torch.rand(shape).cuda(0)
    Word2vec_Model = Word2Vec.load("/home/ao_ding/expand/trace/code/word2vec_model.model")
    sentences = [sent for sent in tokenize(sentense)]
    for a_sentence in sentences:
        context_embeddings = sentence_embedding(Word2vec_Model, a_sentence)
        #print(embedding.size())
        context_embeddings=torch.tensor(context_embeddings).cuda(0)
        context_embeddings = context_embeddings.unsqueeze(0).unsqueeze(1)
        #print(torch.tensor(context_embeddings).size())
        embedding = torch.cat([embedding,torch.tensor(context_embeddings).cuda(0)],dim=1)
    len = embedding.size()[1]
    print(len)
    length_list.append(len)
length_list.sort()
print(length_list[-10:-1])
