import torch
import get_dataset as gds
import os
import sqlite3
from pl_model import Codebert_LSTM
import pytorch_lightning as pl
from transformers import AutoModel, AutoTokenizer 
import argparse


tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
Codebert_model = AutoModel.from_pretrained("microsoft/codebert-base")
embedding_size = 768
max_length = 458
def initdb():
    conn = sqlite3.connect('signature.db')
    c = conn.cursor()
    c.execute("""CREATE TABLE signatures (
            filename TEXT,
            a_signature TEXT,
            label INTEGER
    )""")
    conn.commit()
    conn.close()
    print("init table!")

def generate_signature(PATH=""):
    datalist,labellist,filenamelist = gds.read_dataset()
    model = Codebert_LSTM.load_from_checkpoint(PATH).to(0)
    if(os.path.exists("signature.db")==False):
        initdb()

    conn = sqlite3.connect('signature.db')
    c = conn.cursor()
    
    for sentense,label,filename in zip(datalist,labellist,filenamelist):
        shape=(1,0,embedding_size)
        embedding = torch.rand(shape)
        for a_sentense in sentense:
            nl_tokens=tokenizer.tokenize(a_sentense)
            tokens=[tokenizer.cls_token]+nl_tokens+[tokenizer.sep_token]
            tokens_ids=tokenizer.convert_tokens_to_ids(tokens)
            context_embeddings=Codebert_model(torch.tensor(tokens_ids)[None,:])[0]
            embedding = torch.cat([embedding,context_embeddings],dim=1)
        len = embedding.size()[1]
        if(len<max_length):
            shape=(1,max_length-len,embedding_size)
            zero = torch.zeros(shape)
            embedding = torch.cat([embedding,zero],dim=1)
        #print(self.embedding.shape)
        output=str((model(embedding).tolist())[0])
        c.execute("INSERT INTO signatures VALUES ('%s', '%s', %d)"%(filename,output,label))
        print("creat a signature",output)
        conn.commit()
    conn.close()

if __name__ == "__main__":
    #parser = argparse.ArgumentParser()
    #parser.add_argument('--PATH', type=str, default = None)
    #args=parser.parse_args()
    PATH="/home/ao_ding/gwork/saved/logger_200/checkpoints/sample-epoch=08-val_acc=1.00.ckpt"
    generate_signature(PATH)

    
