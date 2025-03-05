import pandas as pd
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import shutil
import pytorch_lightning as pl
import get_dataset as gds
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer
from pytorch_lightning.loggers import  TensorBoardLogger
from pytorch_lightning.utilities.cloud_io import load as pl_load


tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
Codebert_model = AutoModel.from_pretrained("microsoft/codebert-base")
Codebert_model.cuda(0)
embedding_size = 768
weights = torch.tensor([5240,10648,347,528,240,96], dtype=torch.float32).cuda(0)
weights = weights / weights.sum()
weights = 1.0 / weights
weights = weights / weights.sum()
MAX_LENGTH=643

class CCSDataset:
    """Override getitem for codebert."""

    def __init__(self, partition="train", max_length=643):
        """Init."""
        self.dl = gds.all_dataset(0.8,0.1,0.1)
        self.true_dl = []
        if partition == "train":
            for a_dl in self.dl:
                if(a_dl[2]==0):
                    self.true_dl.append((a_dl[0],a_dl[1]))
                else:
                    continue
        elif partition == "val":
            for a_dl in self.dl:
                if(a_dl[2]==1):
                    self.true_dl.append((a_dl[0],a_dl[1]))
                else:
                    continue
        else:
            for a_dl in self.dl:
                if(a_dl[2]==2):
                    self.true_dl.append((a_dl[0],a_dl[1]))
                else:
                    continue
        self.max_length = max_length

    def __len__(self):
        """Get length of dataset."""
        return len(self.true_dl)

    def __getitem__(self, idx):
        """Override getitem."""
        sentense,self.label = self.true_dl[idx]
        shape=(1,0,embedding_size)
        self.embedding = torch.rand(shape).cuda(0)
        for a_sentense in sentense:
            nl_tokens=tokenizer.tokenize(a_sentense)
            tokens=[tokenizer.cls_token]+nl_tokens+[tokenizer.sep_token]
            tokens_ids=tokenizer.convert_tokens_to_ids(tokens)
            context_embeddings=Codebert_model(torch.tensor(tokens_ids).cuda(0)[None,:])[0].cuda(0)
            self.embedding = torch.cat([self.embedding,context_embeddings],dim=1)
        len = self.embedding.size()[1]
        if(len<self.max_length):
            shape=(1,self.max_length-len,embedding_size)
            zero = torch.zeros(shape).cuda(0)
            self.embedding = torch.cat([self.embedding,zero],dim=1)
        #print(self.embedding.shape)
        self.embedding=self.embedding.reshape(self.max_length,embedding_size)
        #print(self.embedding.shape)
        return self.embedding, self.label

class CCSDatasetDataModule(pl.LightningDataModule):
    """Pytorch Lightning Datamodule for Bigvul."""

    def __init__(self, DataClass, batch_size):
        """Init class from bigvul dataset."""
        super().__init__()
        self.train = DataClass(partition="train",max_length=MAX_LENGTH)
        self.val = DataClass(partition="val",max_length=MAX_LENGTH)
        self.test = DataClass(partition="test",max_length=MAX_LENGTH)
        self.batch_size = batch_size

    def train_dataloader(self):
        """Return train dataloader."""
        return DataLoader(self.train, shuffle=True,num_workers=0 ,batch_size=self.batch_size)

    def val_dataloader(self):
        """Return val dataloader."""
        return DataLoader(self.val,num_workers=0, batch_size=self.batch_size)

    def test_dataloader(self):
        """Return test dataloader."""
        return DataLoader(self.test,num_workers=0, batch_size=self.batch_size)


class Codebert_LSTM(pl.LightningModule):
    """model."""

    def __init__(self, config):
        """Initilisation."""
        super().__init__()
        self.lr = config['lr']
        self.hidden_size = config['hidden_size']
        self.fc1_size = config['fc1_size']
        self.dropout1 = config['dropout1']

        self.save_hyperparameters()
        self.bilstm = nn.LSTM(embedding_size, self.hidden_size, 1,True,False, bidirectional=True)
        self.fc1 = torch.nn.Linear(self.hidden_size*2, self.fc1_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p = self.dropout1)
        self.fc2 = torch.nn.Linear(self.fc1_size, 6)
        self.accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=6)
        self.auroc = torchmetrics.AUROC(task="multiclass", num_classes=6,compute_on_step=False)
        #self.mcc = torchmetrics.MatthewsCorrcoef(2)

    def forward(self, embedding):
        """Forward pass."""
        #print(embedding.shape)
        batch_size = embedding.shape[0]
        input1 = embedding.transpose(0, 1)
        #print(input.shape)
        hidden_state = torch.randn(2*1,batch_size, self.hidden_size).cuda(0)
        cell_state = torch.randn(2*1,batch_size, self.hidden_size).cuda(0)
        output, (h_n, c_n) = self.bilstm(input1,(hidden_state,cell_state))
        # 使用正向LSTM与反向LSTM最后一个输出做拼接
        Bilstmout = torch.cat([h_n[0], h_n[1]], dim=1) # dim=1代表横向拼接
        out = self.fc1(Bilstmout)
        out = self.dropout(out)
        fc1_out = self.relu(out)
        fc2_out = self.fc2(fc1_out)
        return fc2_out

    def training_step(self, batch, batch_idx):
        """Training step."""
        ids, labels = batch
        logits = self(ids)
        loss = F.cross_entropy(logits, labels,weights)
        #loss = FocalLoss(logits,labels)
        #print(loss)
        pred = F.softmax(logits, dim=1)
        acc = self.accuracy(pred.argmax(1), labels)
        #mcc = self.mcc(pred.argmax(1), labels)
        self.log("train_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_acc", acc, prog_bar=True, logger=True)
        #self.log("train_mcc", mcc, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """Validate step."""
        ids, labels = batch
        logits = self(ids)
        loss = F.cross_entropy(logits, labels,weights)
        #loss = FocalLoss(logits,labels)
        pred = F.softmax(logits, dim=1)
        acc = self.accuracy(pred.argmax(1), labels)
        #mcc = self.mcc(pred.argmax(1), labels)

        self.log("val_loss", loss, on_step=True, prog_bar=True, logger=True)
        #self.auroc.update(pred, labels)
        #self.log("val_auroc", self.auroc, prog_bar=True, logger=True)
        self.log("val_acc", acc, prog_bar=True, logger=True)
        #self.log("val_mcc", mcc, prog_bar=True, logger=True)
        return {"loss":loss,"val_acc":acc}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        avg_acc = torch.stack([x["val_acc"] for x in outputs]).mean()
        self.log("ptl/val_loss", avg_loss)
        self.log("ptl/val_accuracy", avg_acc)

    def test_step(self, batch, batch_idx):
        """Test step."""
        ids, labels = batch
        logits = self(ids)
        pred = F.softmax(logits, dim=1)
        loss = F.cross_entropy(logits, labels,weights)
        #loss = FocalLoss(logits,labels)
        self.auroc.update(pred, labels)
        acc = self.accuracy(pred.argmax(1), labels)
        self.log("test_acc", acc, prog_bar=True, logger=True)
        self.log("test_auroc", self.auroc, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        """Configure optimizer."""
        return torch.optim.AdamW(self.parameters(), lr=self.lr)

if __name__=='__main__':
    PATH="/home/ao_ding/gwork/data/D2A/saved/logger_1/checkpoints/sample-epoch=38-val_acc=0.94.ckpt"
    config={}
    batch_size=64
    model = Codebert_LSTM.load_from_checkpoint(PATH)
    model = model.to(0)
    data = CCSDatasetDataModule(CCSDataset,batch_size)
    trainer=pl.Trainer(accelerator='gpu', devices=1)
    trainer.test(model,data)

