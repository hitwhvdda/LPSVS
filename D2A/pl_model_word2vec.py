import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import shutil
import pytorch_lightning as pl
import get_dataset as gds
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
import numpy as np
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer
from pytorch_lightning.loggers import  TensorBoardLogger
# from pytorch_lightning.utilities.cloud_io import load as pl_load
import ray
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler, PopulationBasedTraining
from ray.tune.integration.pytorch_lightning import TuneReportCallback, \
    TuneReportCheckpointCallback
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize


tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
Codebert_model = AutoModel.from_pretrained("microsoft/codebert-base")
Codebert_model.cuda(0)
embedding_size = 300
weights = torch.tensor([5240,10648,347,528,240,96], dtype=torch.float32).cuda(0)
weights = weights / weights.sum()
weights = 1.0 / weights
weights = weights / weights.sum()
MAX_LENGTH=41
Word2vec_Model = Word2Vec.load("/home/ao_ding/expand/D2A/code/word2vec_model.model")
def tokenize(sentences):
    for line in sentences:
        yield word_tokenize(line.strip())

def sentence_embedding(model, sentence):
    return np.mean([model.wv[word] for word in sentence if word in model.wv], axis=0)

class CCSDataset:
    """Override getitem for codebert."""

    def __init__(self, partition="train", max_length=41):
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
        
        sentences = [sent for sent in tokenize(sentense)]
        for a_sentence in sentences:
            context_embeddings = sentence_embedding(Word2vec_Model, a_sentence)
            context_embeddings=torch.tensor(context_embeddings).cuda(0)
            context_embeddings = context_embeddings.unsqueeze(0).unsqueeze(1)
            #print(context_embeddings)
            if(context_embeddings.dim()!=3):
                print(context_embeddings)
                print(a_sentence)
                continue
            self.embedding = torch.cat([self.embedding,context_embeddings],dim=1)
        length = self.embedding.size()[1]
        if(length<self.max_length):
            shape=(1,self.max_length-length,embedding_size)
            zero = torch.zeros(shape).cuda(0)
            self.embedding = torch.cat([self.embedding,zero],dim=1)
        #print(self.embedding.shape)
        self.embedding=self.embedding.reshape(self.max_length,embedding_size)
        #print(self.embedding.shape)
        return self.embedding, self.label

class CCSDatasetDataModule(pl.LightningDataModule):
    """Pytorch Lightning Datamodule for Bigvul."""

    def __init__(self, DataClass, config):
        """Init class from bigvul dataset."""
        super().__init__()
        self.train = DataClass(partition="train",max_length=MAX_LENGTH)
        self.val = DataClass(partition="val",max_length=MAX_LENGTH)
        self.test = DataClass(partition="test",max_length=MAX_LENGTH)
        self.batch_size = config["batch_size"]

    def train_dataloader(self):
        """Return train dataloader."""
        return DataLoader(self.train, shuffle=True,num_workers=0 ,batch_size=self.batch_size)

    def val_dataloader(self):
        """Return val dataloader."""
        return DataLoader(self.val,num_workers=0, batch_size=self.batch_size)

    def test_dataloader(self):
        """Return test dataloader."""
        return DataLoader(self.test,num_workers=0, batch_size=self.batch_size)

class Codebert_BiLSTM(pl.LightningModule):
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
        #self.auroc = torchmetrics.AUROC(task="multiclass", num_classes=6,compute_on_step=False)
        #self.mcc = torchmetrics.MatthewsCorrcoef(2)

    def forward(self, embedding):
        """Forward pass."""
        #print(embedding.shape)
        batch_size = embedding.shape[0]
        input = embedding.transpose(0, 1).cuda(0)
        #print(input.shape)
        hidden_state = torch.randn(2*1,batch_size, self.hidden_size).cuda(0)
        cell_state = torch.randn(2*1,batch_size, self.hidden_size).cuda(0)
        output, (h_n, c_n) = self.bilstm(input,(hidden_state,cell_state))
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
        self.log("train_loss", loss, on_epoch=True, prog_bar=False, logger=True)
        self.log("train_acc", acc,on_epoch=True, prog_bar=True, logger=True)
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

        self.log("val_loss", loss, on_step=True, prog_bar=False, logger=True)
        #self.auroc.update(pred, labels)
        #self.log("val_auroc", self.auroc, prog_bar=True, logger=True)
        self.log("val_acc", acc, on_epoch=True, prog_bar=True, logger=True)
        #self.log("val_mcc", mcc, prog_bar=True, logger=True)


    def test_step(self, batch, batch_idx):
        """Test step."""
        ids, labels = batch
        logits = self(ids)
        pred = F.softmax(logits, dim=1)
        loss = F.cross_entropy(logits, labels,weights)
        #loss = FocalLoss(logits,labels)
        #self.auroc.update(pred, labels)
        acc = self.accuracy(pred.argmax(1), labels)
        self.log("test_acc", acc, prog_bar=True, logger=True)
        #self.log("test_auroc", self.auroc, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        """Configure optimizer."""
        return torch.optim.AdamW(self.parameters(), lr=self.lr)


class Word2vec_BGRU(pl.LightningModule):
    """model."""

    def __init__(self, config):
        """Initilisation."""
        super().__init__()
        self.lr = config['lr']
        self.hidden_size = config['hidden_size']
        self.fc1_size = config['fc1_size']
        self.dropout1 = config['dropout1']

        self.save_hyperparameters()
        self.bgru = nn.GRU(embedding_size, self.hidden_size, 1,batch_first=False, bidirectional=True)
        self.fc1 = torch.nn.Linear(self.hidden_size*2, self.fc1_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p = self.dropout1)
        self.fc2 = torch.nn.Linear(self.fc1_size, 6)
        self.accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=6)
        #self.auroc = torchmetrics.AUROC(task="multiclass", num_classes=6,compute_on_step=False)
        #self.mcc = torchmetrics.MatthewsCorrcoef(2)

    def forward(self, embedding):
        """Forward pass."""
        #print(embedding.shape)
        batch_size = embedding.shape[0]
        input = embedding.transpose(0, 1).cuda(0)
        #print(input.shape)
        output, h_n = self.bgru(input)
        # 使用正向LSTM与反向LSTM最后一个输出做拼接
        bgru_out = torch.cat([h_n[0], h_n[1]], dim=1) # dim=1代表横向拼接
        out = self.fc1(bgru_out)
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
        self.log("train_acc", acc,on_epoch=True, prog_bar=True, logger=True)
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
        self.log("val_acc", acc, on_epoch=True, prog_bar=True, logger=True)
        #self.log("val_mcc", mcc, prog_bar=True, logger=True)


    def test_step(self, batch, batch_idx):
        """Test step."""
        ids, labels = batch
        logits = self(ids)
        pred = F.softmax(logits, dim=1)
        loss = F.cross_entropy(logits, labels,weights)
        #loss = FocalLoss(logits,labels)
        #self.auroc.update(pred, labels)
        acc = self.accuracy(pred.argmax(1), labels)
        self.log("test_acc", acc, prog_bar=True, logger=True)
        #self.log("test_auroc", self.auroc, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        """Configure optimizer."""
        return torch.optim.AdamW(self.parameters(), lr=self.lr)

class Word2vec_BRNN(pl.LightningModule):
    """model."""

    def __init__(self, config):
        """Initilisation."""
        super().__init__()
        self.lr = config['lr']
        self.hidden_size = config['hidden_size']
        self.fc1_size = config['fc1_size']
        self.dropout1 = config['dropout1']

        self.save_hyperparameters()
        self.brnn = nn.RNN(embedding_size, self.hidden_size, 1,batch_first=False, bidirectional=True)
        self.fc1 = torch.nn.Linear(self.hidden_size*2, self.fc1_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p = self.dropout1)
        self.fc2 = torch.nn.Linear(self.fc1_size, 6)
        self.accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=6)
        #self.auroc = torchmetrics.AUROC(task="multiclass", num_classes=6,compute_on_step=False)
        #self.mcc = torchmetrics.MatthewsCorrcoef(2)

    def forward(self, embedding):
        """Forward pass."""
        #print(embedding.shape)
        batch_size = embedding.shape[0]
        input = embedding.transpose(0, 1).cuda(0)
        #print(input.shape)
        output, h_n = self.brnn(input)
        # 使用正向LSTM与反向LSTM最后一个输出做拼接
        bgru_out = torch.cat([h_n[0], h_n[1]], dim=1) # dim=1代表横向拼接
        out = self.fc1(bgru_out)
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
        self.log("train_acc", acc,on_epoch=True, prog_bar=True, logger=True)
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
        self.log("val_acc", acc, on_epoch=True, prog_bar=True, logger=True)
        #self.log("val_mcc", mcc, prog_bar=True, logger=True)


    def test_step(self, batch, batch_idx):
        """Test step."""
        ids, labels = batch
        logits = self(ids)
        pred = F.softmax(logits, dim=1)
        loss = F.cross_entropy(logits, labels,weights)
        #loss = FocalLoss(logits,labels)
        #self.auroc.update(pred, labels)
        acc = self.accuracy(pred.argmax(1), labels)
        self.log("test_acc", acc, prog_bar=True, logger=True)
        #self.log("test_auroc", self.auroc, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        """Configure optimizer."""
        return torch.optim.AdamW(self.parameters(), lr=self.lr)

class Codebert_Transformer(pl.LightningModule):
    """Transformer model class."""

    def __init__(self, config):
        """Initialize the model with config parameters."""
        super().__init__()
        self.lr = config['lr']
        self.hidden_size = config['hidden_size']
        self.fc1_size = config['fc1_size']
        self.dropout1 = config['dropout1']
        self.num_heads = 2
        self.num_layers = 4

        self.save_hyperparameters()
        
        # Transformer model
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_size,
            nhead=self.num_heads,
            dim_feedforward=self.hidden_size,
            dropout=0.1,
            batch_first=False
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=self.num_layers
        )
        self.fc1 = nn.Linear(embedding_size, self.fc1_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=self.dropout1)
        self.fc2 = nn.Linear(self.fc1_size, 6)
        self.accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=7)

    def attention_pooling(seq_output):
        attention_weights = F.softmax(seq_output, dim=0)
        pooled_output = (attention_weights * seq_output).sum(dim=0)
        return pooled_output

    def forward(self, embedding):
        """Forward pass through the model."""
        batch_size = embedding.shape[0]
        input = embedding.transpose(0, 1).cuda(0)
        #print("input",input.shape)
        output = self.transformer_encoder(input)
        #print("output",output.shape)
        output= output[-1]
        #output = output.mean(dim=0)
        #pooled_output = attention_pooling(output)
        #print("output",output.shape)
        out = self.fc1(output)
        out = self.dropout(out)
        fc1_out = self.relu(out)
        fc2_out = self.fc2(fc1_out)
        return fc2_out

    def training_step(self, batch, batch_idx):
        """Training step."""
        ids, labels = batch
        logits = self(ids)
        loss = F.cross_entropy(logits, labels,weights)
        pred = F.softmax(logits, dim=1)
        acc = self.accuracy(pred.argmax(1), labels)
        self.log("train_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_acc", acc, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """Validate step."""
        ids, labels = batch
        logits = self(ids)
        loss = F.cross_entropy(logits, labels,weights)
        pred = F.softmax(logits, dim=1)
        acc = self.accuracy(pred.argmax(1), labels)
        self.log("val_loss", loss, on_step=True, prog_bar=True, logger=True)
        self.log("val_acc", acc, prog_bar=True, logger=True)

    def test_step(self, batch, batch_idx):
        """Test step."""
        ids, labels = batch
        logits = self(ids)
        pred = F.softmax(logits, dim=1)
        loss = F.cross_entropy(logits, labels,weights)
        acc = self.accuracy(pred.argmax(1), labels)
        self.log("test_acc", acc, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        """Configure optimizer."""
        return torch.optim.AdamW(self.parameters(), lr=self.lr)

class Codebert_CNN(pl.LightningModule):
    """CNN model class."""

    def __init__(self, config):
        """Initialize the model with config parameters."""
        super().__init__()
        self.lr = config['lr']
        self.fc1_size = config['fc1_size']
        self.dropout1 = config['dropout1']
        self.num_filters = config['hidden_size']
        self.kernel_sizes = [3, 4, 5]
        self.save_hyperparameters()
        
        # Transformer model
        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels=1, out_channels=self.num_filters, kernel_size=(K, embedding_size)) for K in self.kernel_sizes
        ])
        self.fc1 = nn.Linear(len(self.kernel_sizes) * self.num_filters, self.fc1_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=self.dropout1)
        self.fc2 = nn.Linear(self.fc1_size, 6)
        self.accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=7)

    def forward(self, embedding):
        """Forward pass through the model."""
        input = embedding.unsqueeze(1)
        #print("input",input.shape)
        conved_outputs = [conv(input).squeeze(3) for conv in self.convs]  # [batch_size, num_filters, sequence_length - kernel_size + 1]
        pooled_outputs = [F.max_pool1d(conv_output, conv_output.size(2)).squeeze(2) for conv_output in conved_outputs]  # [batch_size, num_filters]
 
        # 将所有特征组合起来
        cat_output = torch.cat(pooled_outputs, 1)
        #output = output.mean(dim=0)
        #pooled_output = attention_pooling(output)
        #print("cat_output",cat_output.shape)
        out = self.fc1(cat_output)
        out = self.dropout(out)
        fc1_out = self.relu(out)
        fc2_out = self.fc2(fc1_out)
        return fc2_out

    def training_step(self, batch, batch_idx):
        """Training step."""
        ids, labels = batch
        logits = self(ids)
        loss = F.cross_entropy(logits, labels,weights)
        pred = F.softmax(logits, dim=1)
        acc = self.accuracy(pred.argmax(1), labels)
        self.log("train_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_acc", acc, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """Validate step."""
        ids, labels = batch
        logits = self(ids)
        loss = F.cross_entropy(logits, labels,weights)
        pred = F.softmax(logits, dim=1)
        acc = self.accuracy(pred.argmax(1), labels)
        self.log("val_loss", loss, on_step=True, prog_bar=True, logger=True)
        self.log("val_acc", acc, prog_bar=True, logger=True)

    def test_step(self, batch, batch_idx):
        """Test step."""
        ids, labels = batch
        logits = self(ids)
        pred = F.softmax(logits, dim=1)
        loss = F.cross_entropy(logits, labels,weights)
        acc = self.accuracy(pred.argmax(1), labels)
        self.log("test_acc", acc, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        """Configure optimizer."""
        return torch.optim.AdamW(self.parameters(), lr=self.lr)
    
def train_test(config):
    model = Codebert_BiLSTM(config)
    data = CCSDatasetDataModule(CCSDataset,config)
    trainer = pl.Trainer(max_epochs=2,accelerator='gpu',devices=1,enable_progress_bar=True)
    trainer.fit(model,data)
    trainer.test(model,data)

def train_no_tune():
    config = {
        "hidden_size": 768,
        "fc1_size": 256,
        "lr": 1e-3,
        "batch_size": 64,
        "dropout1":0.5
    }
    train_test(config)
# CCSDataset_idx=ray.put(CCSDataset)
# Codebert_LSTM_idx=ray.put(Codebert_LSTM)
# CCSDatasetDataModule_idx=ray.put(CCSDatasetDataModule)
# @ray.remote
# def train_tune(Codebert_LSTM_idx,CCSDataset_idx,CCSDatasetDataModule_idx,config, num_epochs=10, num_gpus=0,is_debug=False):
#     #model = Codebert_LSTM(config)
#     #data = CCSDatasetDataModule(CCSDataset,config)
#     model = Codebert_LSTM_idx(config)
#     data = CCSDatasetDataModule_idx(CCSDataset_idx,config)
#     trainer = pl.Trainer(
#         max_epochs=num_epochs,
#         gpus=num_gpus,
#         fast_dev_run=is_debug,
#         logger=TensorBoardLogger(
#             save_dir=tune.get_trial_dir(), name="", version="."),
#         progress_bar_refresh_rate=0,
#         callbacks=[
#             TuneReportCallback(
#                 {
#                     "train_loss":"train_loss",
#                     "train_acc":"train_acc",
#                     "val_loss": "val_loss",
#                     "val_accuracy": "val_accuracy",
#                     "test_acc":"test_acc"
#                 },
#                 on="test_end"),
#             ModelCheckpoint(
#                 monitor='val_accuracy',
#                 dir_path = "",
#                 filename='sample-{epoch:02d}-{val_acc:.2f}-'+config["lr"]+'-'+config["batch_size"]+'-'+config["hidden_size"]+'-'+config["fc1_size"]+'-'+config["dropout1"]+'-'+config["dropout2"],
#                 verbose = 1,
#                 save_top_k=3,
#                 mode='max',
#                 save_last=True)
#         ])
#     trainer.fit(model,data)
#     trainer.test(model,data)

# def tune_asha(num_samples=20, num_epochs=50, gpus_per_trial=1,is_debug=False):

#     config = {
#         "hidden_size":tune.choice([512,768]),
#         "fc1_size": tune.choice([128, 256, 512]),
#         "dropout1": tune.choice([0.2, 0.3, 0.4,0.5,0.6]),
#         "dropout2": tune.choice([0.3,0.4,0.5,0.6]),
#         "lr": tune.loguniform(1e-4, 1e-1),
#         "batch_size": tune.choice([32, 64, 128,256]),
#     }

#     scheduler = ASHAScheduler(
#         max_t=num_epochs,
#         grace_period=1,
#         reduction_factor=2)

#     reporter = CLIReporter(
#         parameter_columns=["hidden_size","fc1_size","dropout1", "dropout2", "lr", "batch_size"],
#         metric_columns=["train_loss","train_acc","val_loss", "val_accuracy", "test_acc","training_iteration"])

#     analysis = tune.run(
#         tune.with_parameters(
#             ray.get(train_tune.remote()),
#             Codebert_LSTM_idx=Codebert_LSTM_idx,
#             CCSDataset_idx=CCSDataset_idx,
#             CCSDatasetDataModule_idx=CCSDatasetDataModule_idx,
#             num_epochs=num_epochs,
#             num_gpus=gpus_per_trial,
#             is_debug=is_debug),
#         resources_per_trial={
#             "cpu": 2,
#             "gpu": gpus_per_trial
#         },
#         metric="test_loss",
#         mode="min",
#         config=config,
#         num_samples=num_samples,
#         scheduler=scheduler,
#         progress_reporter=reporter,
#         name="tune_asha")

#     print("Best hyperparameters found were: ", analysis.best_config)
def train_and_save(hidden_size,fc1_size,lr,dropout1,batch_size,max_epoch=40):
    config={}
    config["hidden_size"]=hidden_size
    config["fc1_size"]=fc1_size
    config["lr"]=lr
    config["dropout1"]=dropout1
    config["batch_size"]=batch_size
    model = Word2vec_BRNN(config).to(0)
    data = CCSDatasetDataModule(CCSDataset,config)
    logger=TensorBoardLogger(save_dir="/home/ao_ding/expand/D2A/result", name="Word2vec_BRNN", version=".")
    checkpoint_callback = ModelCheckpoint(
    monitor='val_acc',
    filename='sample-{epoch:02d}-{val_acc:.2f}',
    save_top_k=3,
    mode='max',
    save_last=True
)
    #early_stopping = EarlyStopping('val_loss',mode="min",patience=6)
    #printtable=PrintTableMetricsCallback()
    trainer = pl.Trainer(
        default_root_dir="/home/ao_ding/expand/D2A/result",
        max_epochs=max_epoch,
        accelerator='gpu',
        devices=1,
        val_check_interval=0.25,
        precision=16,
        logger=logger,
        enable_progress_bar=True,
        callbacks=[checkpoint_callback]
        )
    trainer.fit(model,data)
    trainer.test(model,data)
    

if __name__=='__main__':
    #train_no_tune()
    
    train_and_save(256,128,0.0001,0.3,64)
