import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import random
from sklearn.ensemble import RandomForestClassifier
from torch.nn.utils.rnn import pad_sequence

dirlist = ["BufferOverflow", "CmdInjection", "DoubleFree", "FormatString", "IntOverflow","IntUnderflow","UseAfterFree"]
#dirlist = ["BufferOverflow"]
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# 使用 GPU 如果可用，否则使用 CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 自定义数据集类
class CFileDataset(Dataset):
    def __init__(self, data_folder):
        self.data_folder = data_folder
        self.files = []
        self.labels = []
        for dir in dirlist:
            workdir = os.path.join(data_folder, dir)
            #print("workdir:",workdir)
            trueworklist = os.listdir(workdir)
            for truework in trueworklist:
                filedir = os.path.join(workdir,truework)
                #print("filedir:",filedir)
                filelist = os.listdir(filedir)
                for file in filelist:
                    #print("append:",os.path.join(filedir, file))
                    self.files.append(os.path.join(filedir, file))
                    self.labels.append(dirlist.index(dir))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        filepath = self.files[idx]
        label = self.labels[idx]

        # 模拟词法分析，这里简单使用分词代替
        with open(filepath, 'r') as file:
            code = file.read()
        tokens = code.split()

        return tokens, label

# 词汇表建立
def build_vocab(data_folder):
    vocab = set()
    for dir in dirlist:
        workdir = os.path.join(data_folder, dir)
        trueworklist = os.listdir(workdir)
        for truework in trueworklist:
            filedir = os.path.join(workdir,truework)
            filelist = os.listdir(filedir)
            for file in filelist:
                with open(os.path.join(filedir, file), 'r') as f:
                    tokens = f.read().split()
                    vocab.update(tokens)
    return {word: idx for idx, word in enumerate(vocab)}

# Embedding层
class CustomEmbedding(nn.Module):
    def __init__(self, vocab_size, embed_size=13):
        super(CustomEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.gaussian_noise = torch.distributions.Normal(0, 0.1)

    def forward(self, x):
        embeds = self.embedding(x)
        noise = self.gaussian_noise.sample(embeds.shape).to(embeds.device)
        return torch.clamp(embeds + noise, -1, 1)

# 卷积+GRU特征提取网络
class ConvRNNModel(nn.Module):
    def __init__(self, vocab_size, embed_size=13, num_classes=10):
        super(ConvRNNModel, self).__init__()
        self.embedding = CustomEmbedding(vocab_size, embed_size)

        # 卷积层
        self.conv = nn.Conv2d(1, 512, (9, embed_size))
        self.bn = nn.BatchNorm2d(512)
        self.relu = nn.ReLU()

        # 双层GRU
        self.gru = nn.GRU(embed_size, 256, num_layers=2, batch_first=True, bidirectional=True)

        # 全连接层
        self.fc1 = nn.Linear(512 + 256 * 2, 64)
        self.fc2 = nn.Linear(64, 16)
        self.fc3 = nn.Linear(16, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # Embedding操作
        x_embed = self.embedding(x)

        # 卷积操作，处理embedding以外的特征
        x_conv = x_embed.unsqueeze(1)  # 增加一个channel维度
        x_conv = self.conv(x_conv)
        x_conv = self.bn(x_conv)
        x_conv = self.relu(x_conv)
        x_conv = x_conv.squeeze(3)  # 去掉最后一个无关维度

        # GRU操作，直接处理embedding
        x_gru, _ = self.gru(x_embed)

        # 池化
        x_pool = F.max_pool1d(x_conv, kernel_size=x_conv.size(2)).squeeze(2)
        x_gru_pool = F.max_pool1d(x_gru.permute(0, 2, 1), kernel_size=x_gru.size(1)).squeeze(2)

        # 拼接卷积和GRU的输出
        x_concat = torch.cat((x_pool, x_gru_pool), dim=1)

        # 全连接层
        x = F.relu(self.fc1(x_concat))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x

# 自定义的collate_fn用于处理不同长度的序列
def collate_fn(batch):
    tokens, labels = zip(*batch)

    # 将token转化为索引并进行padding
    token_indices = [[vocab.get(token, 0) for token in token_list] for token_list in tokens]
    token_indices_padded = pad_sequence([torch.tensor(t) for t in token_indices], batch_first=True, padding_value=0)

    # 转换为Tensor
    labels = torch.tensor(labels, dtype=torch.long)

    return token_indices_padded, labels

# 数据集划分
def split_dataset(dataset, train_ratio=0.8, val_ratio=0.1):
    train_size = int(train_ratio * len(dataset))
    val_size = int(val_ratio * len(dataset))
    test_size = len(dataset) - train_size - val_size
    return random_split(dataset, [train_size, val_size, test_size])

def train_model(model, train_loader, val_loader, test_loader, criterion, optimizer, num_epochs=10):
    model = model.to(device)
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            labels = labels.to(torch.long)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # 验证集上进行评估
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                labels = labels.to(torch.long)

                outputs = model(inputs)
                val_loss += criterion(outputs, labels).item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_loss /= len(val_loader)
        val_accuracy = correct / total
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(train_loader)}, Val Loss: {val_loss}, Val Accuracy: {val_accuracy * 100:.2f}%')

    # 测试集上进行评估
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            labels = labels.to(torch.long)

            outputs = model(inputs)
            test_loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    test_loss /= len(test_loader)
    test_accuracy = correct / total
    print(f'Test Loss: {test_loss}, Test Accuracy: {test_accuracy * 100:.2f}%')

# 加载数据
data_folder = "/home/ao_ding/expand/Compare/data/trace/source_code"
dataset = CFileDataset(data_folder)
vocab = build_vocab(data_folder)

# 数据集划分为训练集、验证集和测试集
train_dataset, val_dataset, test_dataset = split_dataset(dataset)

# 数据加载器
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, collate_fn=collate_fn)

# 初始化模型
model = ConvRNNModel(vocab_size=len(vocab), num_classes=7).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)



# 获取特征并使用随机森林分类器
def extract_features_and_train_rf(model, loader):
    features = []
    labels = []
    model.eval()
    with torch.no_grad():
        for inputs, label in loader:
            inputs = inputs.to(device)
            output = model(inputs)
            features.append(output.detach().cpu().numpy())
            labels.append(label)

    features = np.vstack(features)
    labels = np.hstack(labels)

    rf_classifier = RandomForestClassifier(n_estimators=100)
    rf_classifier.fit(features, labels)

    return rf_classifier

# 在测试集上使用随机森林分类器进行评估
def evaluate_rf_classifier(rf_classifier, model, test_loader):
    features = []
    labels = []
    maxpred = []
    model.eval()
    thresholdlist=[0.3,0.4,0.5,0.6,0.7,0.8,0.85,0.9,0.95,0.98,0.99]
    with torch.no_grad():
        for inputs, label in test_loader:
            inputs = inputs.to(device)
            output = model(inputs)
            features.append(output.detach().cpu().numpy())
            preds = torch.max(F.softmax(output,dim=1)).cpu().numpy().item()
            maxpred.append(preds)
            labels.append(label)

    features = np.vstack(features)
    labels = np.hstack(labels)
    maxpred = np.hstack(maxpred)

    accuracy = rf_classifier.score(features, labels)
    predicted_labels = rf_classifier.predict(features)
    print("features:",features[:2])
    print("labels:",labels[:2])
    print("maxpred:",maxpred[:2])
    print("predicted_labels:",predicted_labels[:2])
    for threshold in thresholdlist:
        tpnum=0
        fpnum=0
        fnnum=0
        for i in range(len(maxpred)):
            pred = predicted_labels[i]
            label = labels[i]
            max = maxpred[i]
            if(max>threshold and pred==label):
                tpnum+=1
            elif(max>threshold and pred!=label):
                fpnum+=1
            else:
                fnnum+=1
        print("threshold:",threshold)
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
    print(f'Random Forest Test Accuracy: {accuracy * 100:.2f}%')

train_model(model, train_loader, val_loader, test_loader, criterion, optimizer,100)
# 提取特征并训练随机森林分类器
rf_classifier = extract_features_and_train_rf(model, train_loader)

# 在测试集上评估随机森林分类器
evaluate_rf_classifier(rf_classifier, model, test_loader)