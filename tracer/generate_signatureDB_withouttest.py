import torch
import get_dataset as gds
import os
import sqlite3
from pl_model import Codebert_BiLSTM,Codebert_BGRU,Codebert_BRNN,Codebert_CNN,Codebert_Transformer
from pl_model_word2vec import Word2vec_BiLSTM
from pl_model_glove import Glove_BiLSTM
from pl_model_fastText import FastText_BiLSTM
import pytorch_lightning as pl
from transformers import AutoModel, AutoTokenizer
from sklearn.model_selection import train_test_split
import argparse
from gensim.models import KeyedVectors,FastText,Word2Vec
from nltk.tokenize import word_tokenize
import numpy as np
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
Codebert_model = AutoModel.from_pretrained("microsoft/codebert-base").cuda(0)
Word2vec_Model = Word2Vec.load("/home/ao_ding/expand/trace/code/word2vec_model.model")
glove_file = "/home/ao_ding/expand/trace/code/glove.txt"
Glove_Model = KeyedVectors.load_word2vec_format(glove_file)
fasttext_file = "/home/ao_ding/expand/trace/code/fasttext_model.model"
FastText_Model = FastText.load(fasttext_file)
embedding_size = 300
max_length = 29
def tokenize(sentences):
    for line in sentences:
        yield word_tokenize(line.strip())


def sentence_embedding(model, sentence):
    return np.mean([model.wv[word] for word in sentence if word in model.wv], axis=0)

def initdb():
    conn = sqlite3.connect(r'/home/ao_ding/expand/trace/database/signature_Word2vec_BiLSTM.db')
    c = conn.cursor()
    c.execute("""CREATE TABLE signatures (
            filename TEXT,
            a_signature TEXT,
            label INTEGER,
            source TEXT
    )""")
    conn.commit()
    conn.close()
    print("init table!")

def generate_signature(PATH=""):
    datalist,labellist,filenamelist = gds.read_dataset()
    newdatalist=[]
    for data,filename in zip(datalist,filenamelist):
        newdatalist.append((data,filename))
    stratify = labellist
    x_train,x_test,y_train,y_test = train_test_split(
        newdatalist,labellist,test_size=1-0.8,random_state=1,stratify=stratify
    )
    stratify = y_test
    x_valid,x_test,y_valid,y_test = train_test_split(
        x_test,y_test,test_size=0.5,random_state=1,stratify=stratify
    )
    #for i in range(1000):
     #   if(y_test[i]==6):
      #      print(x_test[i])
   #exit()
    datalist=[]
    labellist=[]
    filenamelist=[]
    for newdata,label in zip(x_train,y_train):
        datalist.append(newdata[0])
        labellist.append(label)
        filenamelist.append(newdata[1])
    for newdata,label in zip(x_valid,y_valid):
        datalist.append(newdata[0])
        labellist.append(label)
        filenamelist.append(newdata[1])
    
    # for newdata,label in zip(x_test,y_test):
    #     print(newdata)
    #     print(label)
    #     print("-------")

    #model = Codebert_BiLSTM.load_from_checkpoint(PATH).cuda(0)
    #model = Codebert_BGRU.load_from_checkpoint(PATH).cuda(0)
    model = Codebert_BRNN.load_from_checkpoint(PATH).cuda(0)
    #model = Codebert_CNN.load_from_checkpoint(PATH).cuda(0)
    #model = Codebert_Transformer.load_from_checkpoint(PATH).cuda(0)
    #model = Word2vec_BiLSTM.load_from_checkpoint(PATH).cuda(0)
    #model = Glove_BiLSTM.load_from_checkpoint(PATH).cuda(0)
    #model = FastText_BiLSTM.load_from_checkpoint(PATH).cuda(0)
    if(os.path.exists(r"/home/ao_ding/expand/trace/database/signature_Word2vec_BiLSTM.db")!=True):
        initdb()

    conn = sqlite3.connect(r'/home/ao_ding/expand/trace/database/signature_Word2vec_BiLSTM.db')
    c = conn.cursor()
    insertdatalist=[]
    # for sentense,label,filename in zip(datalist,labellist,filenamelist):
    #     shape=(1,0,embedding_size)
    #     embedding = torch.rand(shape).cuda(0)
    #     for a_sentense in sentense:
    #         nl_tokens=tokenizer.tokenize(a_sentense)
    #         tokens=[tokenizer.cls_token]+nl_tokens+[tokenizer.sep_token]
    #         tokens_ids=tokenizer.convert_tokens_to_ids(tokens)
    #         context_embeddings=Codebert_model(torch.tensor(tokens_ids).cuda(0)[None,:])[0]
    #         context_embeddings = torch.mean(context_embeddings,dim=1).unsqueeze(1)
    #         embedding = torch.cat([embedding,context_embeddings],dim=1)
    #     length = embedding.size()[1]
    #     if(length<max_length):
    #         shape=(1,max_length-length,embedding_size)
    #         zero = torch.zeros(shape).cuda(0)
    #         embedding = torch.cat([embedding,zero],dim=1)
    #     #print(self.embedding.shape)
    #     output=str((model(embedding).tolist())[0])
    #     insertdatalist.append((filename,output,label,str(sentense)))
    
    for sentense,label,filename in zip(datalist,labellist,filenamelist):
        shape=(1,0,embedding_size)
        embedding = torch.rand(shape).cuda(0)
        sentences = [sent for sent in tokenize(sentense)]
        for a_sentence in sentences:
            context_embeddings = sentence_embedding(Word2vec_Model, a_sentence)
            #print(self.embedding.size())
            context_embeddings=torch.tensor(context_embeddings).cuda(0)
            context_embeddings = context_embeddings.unsqueeze(0).unsqueeze(1)
            #print(torch.tensor(context_embeddings).size())
            embedding = torch.cat([embedding,context_embeddings],dim=1)
        length = embedding.size()[1]
        if(length<max_length):
            #print(len)
            shape=(1,max_length-length,embedding_size)
            zero = torch.zeros(shape).cuda(0)
            embedding = torch.cat([embedding,zero],dim=1)
        #print(self.embedding.shape)
        output=str((model(embedding).tolist())[0])
        insertdatalist.append((filename,output,label,str(sentense)))
    for x in insertdatalist:
        c.execute("INSERT INTO signatures (filename,a_signature,label,source) VALUES (?,?,?,?)",x)
        print("creat a signature",x)

    conn.commit()
    print("commit success")
    conn.close()
    listlength=len(insertdatalist)
    print(listlength)

if __name__ == "__main__":
    #parser = argparse.ArgumentParser()
    #parser.add_argument('--PATH', type=str, default = None)
    #args=parser.parse_args()
    PATH="/home/ao_ding/expand/trace/result/Word2vec_BiLTSM/checkpoints/sample-epoch=19-val_acc=1.00.ckpt"
    generate_signature(PATH)
