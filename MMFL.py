import csv
import os

import torch
import torch.nn as nn
import torchvision.models as models
from datasets import Dataset
from torch.optim import Adam
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Subset
from torchvision import transforms
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR


# 定义本地文字模型（双向LSTM）
class LocalTextBiLSTM(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size1, hidden_size2, hidden_size_fc, num_classes=6):
        super(LocalTextBiLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm1 = nn.LSTM(embed_size, hidden_size1, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(p=0.3)
        self.lstm2 = nn.LSTM(hidden_size1 * 2, hidden_size2, batch_first=True)
        self.fc1 = nn.Linear(hidden_size2, hidden_size_fc)
        self.fc2 = nn.Linear(hidden_size_fc, num_classes)

    def forward(self, x):
        x = self.embedding(x.long())  # 确保输入类型为 long
        x, _ = self.lstm1(x)
        x = self.dropout(x)
        x, _ = self.lstm2(x)
        x = x[:, -1, :]  # 获取最后一个时间步的输出
        feature = F.relu(self.fc1(x))
        out = self.fc2(feature)
        return out


# 定义本地图像模型（ResNet）
class LocalImageResNetModel(nn.Module):
    def __init__(self):
        super(LocalImageResNetModel, self).__init__()
        self.resnet = models.resnet18(pretrained=True)
        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(in_features, 6)

    def forward(self, x):
        return self.resnet(x)


# 全局模型，用于保存和更新图像与文本嵌入
class GlobalModel(nn.Module):
    def __init__(self):
        super(GlobalModel, self).__init__()
        self.global_text_embeddings = {}
        self.global_image_embeddings = {}

    def update_global_embeddings(self, text_embeddings, image_embeddings, index):
        self.global_text_embeddings[index] = text_embeddings
        self.global_image_embeddings[index] = image_embeddings

    def contrastive_loss(self, local_image_embedding, global_text_embedding, index, margin=1.0):
        # 正样本距离
        positive_distance = F.pairwise_distance(local_image_embedding, global_text_embedding)

        # 负样本距离
        negative_samples = [v for k, v in self.global_text_embeddings.items() if k != index]
        if negative_samples:
            negative_distances = []
            for neg in negative_samples:
                distance = F.pairwise_distance(local_image_embedding, neg)
                negative_distances.append(distance)
            negative_distance = torch.min(torch.cat(negative_distances))
        else:
            negative_distance = torch.tensor(0.0, device=local_image_embedding.device)

        # 对比损失计算
        loss = torch.clamp(margin + positive_distance - negative_distance, min=0.0)
        return loss.mean()

# 蒸馏损失函数：本地模型与全局模型输出之间的KL散度
def distillation_loss(local_output, global_output, temperature):
    local_output = F.log_softmax(local_output / temperature, dim=-1)
    global_output = F.softmax(global_output / temperature, dim=-1)
    return F.kl_div(local_output, global_output, reduction='batchmean') * (temperature ** 2)

# 联邦蒸馏过程
def federated_distillation(local_text_model, local_image_model, global_model, local_train_loader, optimizer_text,
                           optimizer_image, temperature, device):
    local_text_model.train()
    local_image_model.train()
    total_text_loss = 0.0
    total_image_loss = 0.0
    total_samples = 0
    for batch_idx, data in enumerate(tqdm(local_train_loader)):
        bytecode = data['bytecode_vectorizer'].to(device)
        image = data['image'].to(device)
        optimizer_text.zero_grad()
        optimizer_image.zero_grad()
        # 本地模型生成的嵌入
        local_text_embedding = local_text_model(bytecode)
        local_image_embedding = local_image_model(image)
        # 全局模型生成的嵌入（作为软标签）
        global_text_embedding = global_model.global_text_embeddings[batch_idx].to(device)
        global_image_embedding = global_model.global_image_embeddings[batch_idx].to(device)
        # 计算蒸馏损失
        text_loss = distillation_loss(local_text_embedding, global_text_embedding, temperature=temperature)
        image_loss = distillation_loss(local_image_embedding, global_image_embedding, temperature=temperature)
        # 反向传播和优化
        text_loss.backward()
        optimizer_text.step()
        image_loss.backward()
        optimizer_image.step()
        total_text_loss += text_loss.item()
        total_image_loss += image_loss.item()
        total_samples += bytecode.size(0)
        print(f"Federated Distillation - Batch {batch_idx}: Text Loss: {text_loss:.4f}, Image Loss: {image_loss:.4f}")
    avg_text_loss = total_text_loss / total_samples
    avg_image_loss = total_image_loss / total_samples
    print(f"Avg Distillation Loss - Text: {avg_text_loss:.4f}, Image: {avg_image_loss:.4f}")
    return avg_text_loss, avg_image_loss

# 客户端训练和上传步骤
def client_train_and_upload(local_text_model, local_image_model, global_model, data_loader, optimizer_text, optimizer_image, device):
    local_text_model.train()
    local_image_model.train()

    loss_fn = nn.BCEWithLogitsLoss()
    total_text_loss = 0.0
    total_image_loss = 0.0
    total_samples = 0

    for batch_idx, data in enumerate(tqdm(data_loader)):
        bytecode = data['bytecode_vectorizer'].to(device)
        image = data['image'].to(device)
        label = data['byte_label'].to(device)

        optimizer_text.zero_grad()
        optimizer_image.zero_grad()

        text_embedding = local_text_model(bytecode)
        image_embedding = local_image_model(image)

        # 只更新全局模型的嵌入
        global_model.update_global_embeddings(text_embedding.detach(), image_embedding.detach(), batch_idx)

        # 计算局部的 BCE 损失
        text_loss = loss_fn(text_embedding, label)
        image_loss = loss_fn(image_embedding, label)
        total_loss = (text_loss + image_loss) / 2  # 这里只包含 BCE 损失
        total_loss.backward()
        optimizer_text.step()
        optimizer_image.step()

        text_preds = torch.sigmoid(text_embedding).round()
        image_preds = torch.sigmoid(image_embedding).round()

        text_acc = (text_preds.eq(label).sum().item() / label.numel())
        image_acc = (image_preds.eq(label).sum().item() / label.numel())
        print(f"Loss - Image: {image_loss:.4f}, Bytecode: {text_loss:.4f} | Accuracy - Image: {image_acc:.4f}, Bytecode: {text_acc:.4f}")

        total_text_loss += text_loss.item()
        total_image_loss += image_loss.item()
        total_samples += label.numel()

        torch.nn.utils.clip_grad_norm_(local_text_model.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(local_image_model.parameters(), max_norm=1.0)

    avg_text_loss = total_text_loss / total_samples
    avg_image_loss = total_image_loss / total_samples
    return avg_text_loss, avg_image_loss


# 全局服务器计算对比学习损失并进行聚合
def global_update2(global_model, local_models, optimizer_text, optimizer_image, global_train_loader, device):
    local_text_model, local_image_model = local_models
    local_text_model.train()  # 确保文本模型处于训练模式
    local_image_model.train()  # 确保图像模型处于训练模式
    all_global_text_embeddings = torch.stack(list(global_model.global_text_embeddings.values()))  # (338, embedding_dim)
    all_global_image_embeddings = torch.stack(list(global_model.global_image_embeddings.values()))  # (338, embedding_dim)

    for index, batch_data in enumerate(tqdm(global_train_loader)):
        optimizer_text.zero_grad()  # 清空文本模型的梯度
        optimizer_image.zero_grad()  # 清空图像模型的梯度

        local_text_model, local_image_model = local_models
        bytecode = batch_data['bytecode_vectorizer'].to(device)
        image = batch_data['image'].to(device)

        local_text_embedding = local_text_model(bytecode)
        local_image_embedding = local_image_model(image)

        ## 1. 对比学习损失 - 局部文本嵌入和全局图像嵌入（用于更新本地文本模型）
        negative_image_embedding = []
        for local_embedding in local_text_embedding:  # 对每个局部文本嵌入
            distances = F.pairwise_distance(local_embedding.unsqueeze(0), all_global_image_embeddings)  # (338,)
            negative_image_embedding.append(torch.min(distances))
        negative_image_embedding = torch.stack(negative_image_embedding)  # (32,)

        # 计算正样本的全局图像嵌入
        positive_image_embedding = global_model.global_image_embeddings[index].to(device)  # 正样本全局图像嵌入
        positive_distance_text_to_image = F.pairwise_distance(local_text_embedding, positive_image_embedding).to(device)

        # 定义对比损失
        margin = 0.2
        loss_text_image = torch.clamp(margin + positive_distance_text_to_image - negative_image_embedding, min=0.0).mean()

        # 反向传播文本到图像的损失到本地文本模型
        # loss_text_image.backward()
        # optimizer_text.step()

        ## 2. 对比学习损失 - 局部图像嵌入和全局文本嵌入（用于更新本地图像模型）
        negative_text_embedding = []
        for local_embedding in local_image_embedding:  # 对每个局部图像嵌入
            distances = F.pairwise_distance(local_embedding.unsqueeze(0), all_global_text_embeddings)  # (338,)
            negative_text_embedding.append(torch.min(distances))
        negative_text_embedding = torch.stack(negative_text_embedding)  # (32,)

        positive_text_embedding = global_model.global_text_embeddings[index].to(device)  # 正样本全局文本嵌入
        positive_distance_image_to_text = F.pairwise_distance(local_image_embedding, positive_text_embedding).to(device)

        loss_image_text = torch.clamp(margin + positive_distance_image_to_text - negative_text_embedding, min=0.0).mean()

        # 反向传播图像到文本的损失到本地图像模型
        # loss_image_text.backward()
        # optimizer_image.step()

        total_loss = (loss_text_image+loss_image_text)/2
        total_loss.backward()
        optimizer_text.step()
        optimizer_image.step()

        print(f'Global Update - Batch {index}: Contrastive Loss - Text: {loss_text_image:.4f}, Image: {loss_image_text:.4f}')

def global_update(global_model, local_models, optimizer_text, optimizer_image, global_train_loader, device):
    local_text_model, local_image_model = local_models
    local_text_model.train()  # 确保文本模型处于训练模式
    local_image_model.train()  # 确保图像模型处于训练模式
    all_global_text_embeddings = F.normalize(torch.stack(list(global_model.global_text_embeddings.values())), dim=1)  # (338, embedding_dim)
    all_global_image_embeddings = F.normalize(torch.stack(list(global_model.global_image_embeddings.values())), dim=1)  # (338, embedding_dim)

    for index, batch_data in enumerate(tqdm(global_train_loader)):
        optimizer_text.zero_grad()  # 清空文本模型的梯度
        optimizer_image.zero_grad()  # 清空图像模型的梯度

        bytecode = batch_data['bytecode_vectorizer'].to(device)
        image = batch_data['image'].to(device)

        local_text_embedding = F.normalize(local_text_model(bytecode), dim=1)
        local_image_embedding = F.normalize(local_image_model(image), dim=1)

        ## 1. 对比学习损失 - 局部文本嵌入和全局图像嵌入（用于更新本地文本模型）
        # 负样本距离：矩阵运算
        distances_text_to_images = torch.cdist(local_text_embedding, all_global_image_embeddings)  # (batch_size, 338)
        negative_image_embedding = distances_text_to_images.min(dim=1).values  # (batch_size,)

        # 正样本距离
        positive_image_embedding = F.normalize(global_model.global_image_embeddings[index].to(device), dim=1)
        positive_distance_text_to_image = F.pairwise_distance(local_text_embedding, positive_image_embedding)

        # 对比损失
        margin = 0.2
        loss_text_image = torch.clamp(margin + positive_distance_text_to_image - negative_image_embedding, min=0.0).mean()

        ## 2. 对比学习损失 - 局部图像嵌入和全局文本嵌入（用于更新本地图像模型）
        # 负样本距离：矩阵运算
        distances_image_to_texts = torch.cdist(local_image_embedding, all_global_text_embeddings)  # (batch_size, 338)
        negative_text_embedding = distances_image_to_texts.min(dim=1).values  # (batch_size,)

        # 正样本距离
        positive_text_embedding = F.normalize(global_model.global_text_embeddings[index].to(device), dim=1)
        positive_distance_image_to_text = F.pairwise_distance(local_image_embedding, positive_text_embedding)

        # 对比损失
        loss_image_text = torch.clamp(margin + positive_distance_image_to_text - negative_text_embedding, min=0.0).mean()

        # 合并损失并优化
        total_loss = (loss_text_image + loss_image_text) / 2
        total_loss.backward()
        optimizer_text.step()
        optimizer_image.step()

        print(f'Global Update - Batch {index}: Contrastive Loss - Text: {loss_text_image:.4f}, Image: {loss_image_text:.4f}')


# 模型测试
def test_model(local_text_model, local_image_model, test_loader, device):
    local_text_model.eval()
    local_image_model.eval()

    loss_fn = nn.BCEWithLogitsLoss()
    text_total_loss = 0.0
    image_total_loss = 0.0
    text_correct = 0
    image_correct = 0
    total_samples = 0

    # 用于存储每个批次的标签和预测结果
    all_text_preds = []
    all_image_preds = []
    all_labels = []

    with torch.no_grad():
        for batch_idx, data in enumerate(tqdm(test_loader)):
            bytecode = data['bytecode_vectorizer'].to(device)
            image = data['image'].to(device)
            label = data['byte_label'].to(device)

            text_embedding = local_text_model(bytecode)
            image_embedding = local_image_model(image)

            text_loss = loss_fn(text_embedding, label)
            image_loss = loss_fn(image_embedding, label)

            text_total_loss += text_loss.item()
            image_total_loss += image_loss.item()
            total_samples += label.numel()

            text_preds = torch.sigmoid(text_embedding).round()
            image_preds = torch.sigmoid(image_embedding).round()

            text_correct += text_preds.eq(label).sum().item()
            image_correct += image_preds.eq(label).sum().item()

        avg_text_loss = text_total_loss / total_samples
        avg_image_loss = image_total_loss / total_samples
        text_accuracy = text_correct / total_samples
        image_accuracy = image_correct / total_samples
        print(f"Test Loss - Text: {avg_text_loss:.4f}, Image: {avg_image_loss:.4f}")
        print(f"Test Accuracy - Text: {text_accuracy:.4f}, Image: {image_accuracy:.4f}")

        return avg_text_loss, avg_image_loss, text_accuracy, image_accuracy

    #         # 将预测和标签添加到列表中
    #         all_text_preds.extend(text_preds.cpu().numpy())
    #         all_image_preds.extend(image_preds.cpu().numpy())
    #         all_labels.extend(label.cpu().numpy())
    #
    # # 计算平均损失
    # avg_text_loss = text_total_loss / total_samples
    # avg_image_loss = image_total_loss / total_samples
    #
    # # 计算文本模型的指标
    # text_accuracy = accuracy_score(all_labels, all_text_preds)
    # text_precision = precision_score(all_labels, all_text_preds, average='macro')
    # text_recall = recall_score(all_labels, all_text_preds, average='macro')
    # text_f1 = f1_score(all_labels, all_text_preds, average='macro')
    #
    # # 计算图像模型的指标
    # image_accuracy = accuracy_score(all_labels, all_image_preds)
    # image_precision = precision_score(all_labels, all_image_preds, average='macro')
    # image_recall = recall_score(all_labels, all_image_preds, average='macro')
    # image_f1 = f1_score(all_labels, all_image_preds, average='macro')
    #
    # # 创建数据框并格式化输出
    # data = {
    #     "Text Model": [text_accuracy, text_precision, text_recall, text_f1],
    #     "Image Model": [image_accuracy, image_precision, image_recall, image_f1]
    # }
    # metrics_df = pd.DataFrame(data, index=["accuracy", "precision", "recall", "f1"])
    # print(f"Average Test Loss - Text: {avg_text_loss:.4f}, Image: {avg_image_loss:.4f}")
    # print(metrics_df)

# 定义图像增强
# data_transforms = transforms.Compose([
#     transforms.RandomHorizontalFlip(),
#     transforms.RandomRotation(10),
#     transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
#     transforms.ToTensor(),
# ])
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),  # 随机水平翻转
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # ResNet 预训练模型的标准化
])

class CustomDataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, idx):
        data_sample = self.dataset[idx]

        bytecode = torch.tensor(data_sample['bytecode_vectorizer'], dtype=torch.float)

        # 确保 image 为三通道
        image = torch.tensor(data_sample['image'], dtype=torch.float)
        if image.shape[0] == 1:  # 如果图像只有一个通道
            image = image.repeat(3, 1, 1)  # 复制为三个通道

        byte_label = torch.tensor(data_sample['byte_label'], dtype=torch.float)
        image_label = torch.tensor(data_sample['image_label'], dtype=torch.float)

        return {
            'bytecode_vectorizer': bytecode,
            'image': image,
            'byte_label': byte_label,
            'image_label': image_label
        }

    def __len__(self):
        return len(self.dataset)

class EarlyStopping:
    def __init__(self, patience=5, delta=0, save_path="best_model.pt"):
        self.patience = patience
        self.delta = delta
        self.best_score = None
        self.early_stop = False
        self.counter = 0
        self.save_path = save_path  # 模型保存路径

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(model)
            self.counter = 0

    def save_checkpoint(self, model):
        """保存当前模型"""
        torch.save(model.state_dict(), self.save_path)
        print(f"Model saved to {self.save_path}")

def save_test_results(epoch, text_loss, image_loss, text_acc, image_acc, file_path="test_results.csv"):
    # 如果文件不存在，则创建文件并写入表头
    is_new_file = not os.path.exists(file_path)
    with open(file_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        if is_new_file:
            writer.writerow(["Epoch", "Text Loss", "Image Loss", "Text Accuracy", "Image Accuracy"])
        writer.writerow([epoch, text_loss, image_loss, text_acc, image_acc])

# 主训练流程
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    # 数据加载和预处理
    train_ds = torch.load('train_ds.pt')
    val_ds = torch.load('val_ds.pt')
    test_ds = torch.load('test_ds.pt')
    print(train_ds, val_ds, test_ds)

    # 获取数据集的总行数
    total_num_rows = len(train_ds['train'])
    mid_point = total_num_rows // 2  # 数据集的中间点

    # 通过索引切分数据集
    train_subset_indices = list(range(0, mid_point))  # 前一半数据用于训练
    test_subset_indices = list(range(mid_point, total_num_rows))  # 后一半数据用于测试

    # 创建训练和测试数据集
    global_train = Subset(train_ds['train'], train_subset_indices)
    local_train = Subset(train_ds['train'], test_subset_indices)
    print(len(global_train), len(local_train))

    def custom_collate_fn(batch):
        # 确保 bytecode_vectorizer 的形状是 (batch_size, sequence_length, embed_size)
        return {
            'bytecode_vectorizer': torch.stack([item['bytecode_vectorizer'] for item in batch]),
            'image': torch.stack([item['image'] for item in batch]),
            'byte_label': torch.stack([item['byte_label'] for item in batch]),
            'image_label': torch.stack([item['image_label'] for item in batch]),
        }

    batch_size = 64
    global_train_loader = DataLoader(
        CustomDataset(global_train, transform=data_transforms),
        batch_size=batch_size,
        drop_last=True,
        shuffle=True,
        collate_fn=custom_collate_fn)
    local_train_loader = DataLoader(
        CustomDataset(train_ds['train'], transform=data_transforms),
        batch_size=batch_size,
        drop_last=True,
        shuffle=True,
        collate_fn=custom_collate_fn)
    val_loader = DataLoader(
        CustomDataset(val_ds['train'], transform=data_transforms),
        batch_size=batch_size,
        drop_last=True,
        shuffle=True,
        collate_fn=custom_collate_fn)
    test_loader = DataLoader(
        CustomDataset(test_ds['train'], transform=data_transforms),
        batch_size=batch_size,
        drop_last=True,
        shuffle=True,
        collate_fn=custom_collate_fn)

    # 假设字节码向量大小是 500，词嵌入大小是 128，LSTM 隐藏层大小分别是 128 和 64，类别数为 6
    vocab_size = 500  # 假设的字节码向量词汇表大小
    embed_size = 256  # 词嵌入大小
    hidden_size1 = 64  # LSTM 隐藏层1的大小
    hidden_size2 = 64  # LSTM 隐藏层2的大小
    hidden_size_fc = 32  # 全连接层的大小
    epoches = 30

    # 初始化模型
    local_text_model = LocalTextBiLSTM(vocab_size=vocab_size, embed_size=embed_size, hidden_size1=hidden_size1,
                                       hidden_size2=hidden_size2, hidden_size_fc=hidden_size_fc).to(device)
    local_image_model = LocalImageResNetModel().to(device)

    global_model = GlobalModel()

    optimizer_text = Adam(local_text_model.parameters(), lr=1e-3)
    optimizer_image = Adam(local_image_model.parameters(), lr=1e-3)
    scheduler_text = StepLR(optimizer_text, step_size=2, gamma=0.1)  # 每step_size轮学习率衰减为原来的0.1
    scheduler_image = StepLR(optimizer_image, step_size=2, gamma=0.1)

    local_models = [local_text_model, local_image_model]

    # 提前终止机制分别用于文字和图像模型
    early_stopping_text = EarlyStopping(patience=5, delta=0, save_path="checkpoints/best_text_model.pt")
    early_stopping_image = EarlyStopping(patience=5, delta=0, save_path="checkpoints/best_image_model.pt")

    # 初始化温度和衰减参数
    initial_temperature = 5.0
    final_temperature = 1.0
    decay_rate = 0.5  # 每个 epoch 的衰减因子

    for epoch in range(epoches):
        print(f"Epoch {epoch + 1}/{epoches}")

        # 更新温度
        current_temperature = initial_temperature * (decay_rate ** epoch)
        current_temperature = max(current_temperature, final_temperature)
        print(f"Current temperature: {current_temperature}")

        client_train_and_upload(local_text_model, local_image_model, global_model, local_train_loader, optimizer_text,
                                optimizer_image, device)

        # 测试模型
        # val_text_loss, val_image_loss, val_text_acc, val_image_acc = test_model(local_text_model, local_image_model, test_loader, device)

        # 全局更新
        global_update(global_model, local_models, optimizer_text, optimizer_image, global_train_loader, device)

        # 联邦蒸馏（不再需要对本地模型重新训练）
        avg_distillation_loss_text, avg_distillation_loss_image = federated_distillation(local_text_model, local_image_model, global_model, local_train_loader, optimizer_text,
                               optimizer_image, current_temperature, device)

        # 验证模型
        val_text_loss, val_image_loss, val_text_acc, val_image_acc = test_model(local_text_model, local_image_model, val_loader, device)
        print(
            f"Validation - Text Loss: {val_text_loss:.4f}, Image Loss: {val_image_loss:.4f}, Text Acc: {val_text_acc:.4f}, Image Acc: {val_image_acc:.4f}")

        # 提前终止检查
        early_stopping_text(val_text_loss, local_text_model)
        early_stopping_image(val_image_loss, local_image_model)

        # 检查是否应该提前终止
        if early_stopping_text.early_stop and early_stopping_image.early_stop:
            print("Early stopping triggered")
            break

        # 更新学习率
        scheduler_text.step()
        scheduler_image.step()

        text_loss, image_loss, text_acc, image_acc = test_model(local_text_model, local_image_model, test_loader, device)
        # 保存测试结果到文件
        save_test_results(epoch + 1, text_loss, image_loss, text_acc, image_acc,  file_path="test_results.csv")

    # 最终测试时加载最佳模型
    print("Loading the best models for testing...")
    local_text_model.load_state_dict(torch.load("checkpoints/best_text_model.pt"))
    local_image_model.load_state_dict(torch.load("checkpoints/best_image_model.pt"))

    # 最终测试
    avg_text_loss, avg_image_loss, text_accuracy, image_accuracy = test_model(local_text_model, local_image_model, test_loader, device)


if __name__ == "__main__":
    main()