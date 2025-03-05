import torch
import get_dataset as gds
import os
import sqlite3
from pl_model import Codebert_BiLSTM,Codebert_BGRU,Codebert_BRNN,Codebert_CNN,Codebert_Transformer
import pytorch_lightning as pl
from pl_model_word2vec import Word2vec_BGRU
from pl_model_glove import Glove_BGRU
from pl_model_fasttext import FastText_BGRU
from transformers import AutoModel, AutoTokenizer
from sklearn.model_selection import train_test_split
from scipy import spatial
from gensim.models import Word2Vec, KeyedVectors,FastText
from nltk.tokenize import word_tokenize
import numpy as np
import argparse


tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
Codebert_model = AutoModel.from_pretrained("microsoft/codebert-base").cuda(0)
# Word2vec_Model = Word2Vec.load("/home/ao_ding/expand/D2A/code/word2vec_model.model")
# fasttext_file = "/home/ao_ding/expand/D2A/code/fasttext_model.model"
# FastText_Model = FastText.load(fasttext_file)
# glove_file = "/home/ao_ding/expand/D2A/code/glove.txt"
# Glove_Model = KeyedVectors.load_word2vec_format(glove_file)
embedding_size = 768
max_length = 41
list0=[]
list1=[]
list2=[]
list3=[]
list4=[]
list5=[]

listall=[]
thresholdlist=[0.8,0.85,0.9,0.95,0.98,0.99,0.995,0.998,0.999]
def tokenize(sentences):
    for line in sentences:
        yield word_tokenize(line.strip())

def sentence_embedding(model, sentence):
    vectorlist=[]
    for word in sentence:
        vectorlist.append(model.wv[word])
    return np.mean(vectorlist,axis=0)

def selectforlist():
    conn = sqlite3.connect('/home/ao_ding/expand/D2A/database/CodeBERT_Transformer_MEAN4.db')
    global list0,list1,list2,list3,list4,list5,list6,listall
    c = conn.cursor()
    c.execute("SELECT * FROM signatures WHERE label = 0")
    list0=c.fetchall()
    c.execute("SELECT * FROM signatures WHERE label = 1")
    list1=c.fetchall()
    c.execute("SELECT * FROM signatures WHERE label = 2")
    list2=c.fetchall()
    c.execute("SELECT * FROM signatures WHERE label = 3")
    list3=c.fetchall()
    c.execute("SELECT * FROM signatures WHERE label = 4")
    list4=c.fetchall()
    c.execute("SELECT * FROM signatures WHERE label = 5")
    list5=c.fetchall()
    c.execute("SELECT * FROM signatures")
    listall=c.fetchall()
    print(list0[:10],list1[:10],list2[:10],list3[:10],list4[:10],list5[:10])
    conn.close()

def compare(output):
    global list0,list1,list2,list3,list4,list5,listall
    maxsim=0.0
    signature_filename=""
    label_pred=0
    signature_trace = ""
    # if label==0:
    #     list=list0
    # elif label==1:
    #     list=list1
    # elif label==2:
    #     list=list2
    # elif label==3:
    #     list=list3
    # elif label==4:
    #     list=list4
    # elif label==5:
    #     list=list5
    # else:
    #     list=list6
    listlen=len(listall)
    for i in range(listlen):
        a_data=listall[i]
        filename=a_data[0]
        signature=eval(a_data[1])
        label=a_data[2]
        source = eval(a_data[3])
        cos_sim = 1 - spatial.distance.cosine(output, signature)
        if cos_sim>=maxsim:
            signature_filename=filename
            maxsim=cos_sim
            label_pred=label
            signature_trace = source
    signature_source=signature_filename.split('_')[0]
    return signature_source,maxsim,label_pred,signature_trace

def test_withsignature(PATH=""):
    selectforlist()
    datalist,labellist,filenamelist = gds.read_dataset()
    newdatalist=[]
    for data,filename in zip(datalist,filenamelist):
        newdatalist.append([data,filename])
    stratify = labellist
    x_train,x_test,y_train,y_test = train_test_split(
        newdatalist,labellist,test_size=1-0.8,random_state=1,stratify=stratify
    )
    stratify = y_test
    x_valid,x_test,y_valid,y_test = train_test_split(
        x_test,y_test,test_size=0.5,random_state=1,stratify=stratify
    )
    datalist=[]
    labellist=[]
    filenamelist=[]
    for newdata,label in zip(x_test,y_test):
        datalist.append(newdata[0])
        labellist.append(label)
        filenamelist.append(newdata[1])
        if label=="0":
            print(newdata[1],label)

    model = Codebert_Transformer.load_from_checkpoint(PATH).cuda(0)
    #model = Codebert_BiLSTM.load_from_checkpoint(PATH).cuda(0)
    #model = Codebert_BRNN.load_from_checkpoint(PATH).cuda(0)
    #model = Word2vec_BGRU.load_from_checkpoint(PATH).cuda(0)
    #model = Glove_BGRU.load_from_checkpoint(PATH).cuda(0)
    #model = FastText_BGRU.load_from_checkpoint(PATH).cuda(0)
    #model = Codebert_CNN.load_from_checkpoint(PATH).cuda(0)
    fvlist=[]
    for sentense,label,filename in zip(datalist,labellist,filenamelist):
    #for sentense,label in zip(x_test,y_test):
        shape=(1,0,embedding_size)
        embedding = torch.rand(shape).cuda(0)
        for a_sentense in sentense:
            nl_tokens=tokenizer.tokenize(a_sentense)
            tokens=[tokenizer.cls_token]+nl_tokens+[tokenizer.sep_token]
            tokens_ids=tokenizer.convert_tokens_to_ids(tokens)
            context_embeddings=Codebert_model(torch.tensor(tokens_ids).cuda(0)[None,:])[0]
            context_embeddings = torch.mean(context_embeddings,dim=1).unsqueeze(1)
            embedding = torch.cat([embedding,context_embeddings],dim=1)
        embeddinglen = embedding.size()[1]
        if(embeddinglen<max_length):
            shape=(1,max_length-embeddinglen,embedding_size)
            zero = torch.zeros(shape).cuda(0)
            embedding = torch.cat([embedding,zero],dim=1)
        #print(self.embedding.shape)++
        output=(model(embedding).tolist())[0]
        signature_source,cos_sim,label_pred,signature_trace=compare(output)
        fvlist.append((signature_source,cos_sim,label_pred,label,signature_trace))
    # for sentense,label,filename in zip(datalist,labellist,filenamelist):
    #     shape=(1,0,embedding_size)
    #     embedding = torch.rand(shape).cuda(0)
    #     # glove_file = "/home/ao_ding/expand/trace/code/glove.txt"
    #     # Glove_Model = KeyedVectors.load_word2vec_format(glove_file)
    #     #Word2vec_Model = Word2Vec.load("/home/ao_ding/expand/trace/code/word2vec_model.model")
    #     #fasttext_file = "/home/ao_ding/expand/trace/code/fasttext_model.model"
    #     #FastText_Model = FastText.load(fasttext_file)
    #     sentences = [sent for sent in tokenize(sentense)]
    #     for a_sentence in sentences:
    #         context_embeddings = sentence_embedding(FastText_Model, a_sentence)
    #         #print(self.embedding.size())
    #         context_embeddings=torch.tensor(context_embeddings).cuda(0)
    #         context_embeddings = context_embeddings.unsqueeze(0).unsqueeze(1)
    #         #print(torch.tensor(context_embeddings).size())
    #         embedding = torch.cat([embedding,context_embeddings],dim=1)
    #     length = embedding.size()[1]
    #     if(length<max_length):
    #         #print(len)
    #         shape=(1,max_length-length,embedding_size)
    #         zero = torch.zeros(shape).cuda(0)
    #         embedding = torch.cat([embedding,zero],dim=1)
    #     #print(self.embedding.shape)
    #     output=str((model(embedding).tolist())[0])
    #     signature_source,cos_sim,label_pred,signature_trace=compare(output)
    #     fvlist.append((signature_source,cos_sim,label_pred,label,signature_trace))
    for threshold in thresholdlist:
        tplist=[]
        fplist=[]
        fnlist=[]
        for a_fv in fvlist:
            signature_source=a_fv[0]
            cos_sim=a_fv[1]
            label_pred=a_fv[2]
            label=a_fv[3]
            if(cos_sim>threshold and int(label_pred)==int(label)):
                tplist.append((signature_source,cos_sim,label_pred))
            elif(cos_sim>threshold and int(label_pred)!=int(label)):
                fplist.append((signature_source,cos_sim,label_pred))
            else:
                fnlist.append((signature_source,cos_sim,label_pred))

        print(fnlist[:10])
        print("threshold:",threshold)
        tpnum=len(tplist)
        fpnum=len(fplist)
        fnnum=len(fnlist)
        print("tpnum:",tpnum)
        print("fpnum:",fpnum)
        print("fnnum:",fnnum)
        accuracy=tpnum/(tpnum+fpnum+fnnum)
        precision=tpnum/(tpnum+fpnum)
        recall=tpnum/(tpnum+fnnum)
        f1score=2*precision*recall/(precision+recall)
        print("accuracy:",accuracy)
        print("precison:",precision)
        print("recall:",recall)
        print("f1score:",f1score)


if __name__ == "__main__":
    #parser = argparse.ArgumentParser()
    #parser.add_argument('--PATH', type=str, default = None)
    #args=parser.parse_args()
    PATH="/home/ao_ding/expand/D2A/result/CodeBERT_Transformer_MEAN4/checkpoints/sample-epoch=99-val_acc=0.96.ckpt"
    
    test_withsignature(PATH)
