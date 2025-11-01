import os
import time

# 禁用Hugging Face自动连接
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
print("已配置为离线模式，不连接Hugging Face服务器")

import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from torch.utils.data import DataLoader, Dataset
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, get_linear_schedule_with_warmup
import torch.optim as optim
from tqdm import tqdm
import json

# 设置随机种子确保结果可复现
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

set_seed(42)

# 情绪标签定义（六种基本情绪）
emotion_labels = ['happy', 'sad', 'angry', 'fear', 'surprise', 'neutral']
# 中文到英文的情绪映射
chinese_to_english = {
    'happy': 'happy',
    'sad': 'sad', 
    'angry': 'angry',
    'fear': 'fear',
    'surprise': 'surprise',
    'neutral': 'neutral',
    '喜悦': 'happy',
    '悲伤': 'sad',
    '愤怒': 'angry',
    '恐惧': 'fear',
    '惊讶': 'surprise',
    '中性': 'neutral'
}

# 读取数据集
def load_and_preprocess_data(chinese_csv_path, english_csv_path):
    print("正在加载数据集...")
    
    # 读取中文数据集
    chinese_df = pd.read_csv(chinese_csv_path)
    chinese_df['content'] = chinese_df['content'].astype(str)
    # 转换中文标签到英文
    chinese_df['label'] = chinese_df['label'].apply(
        lambda x: chinese_to_english.get(x, x) if isinstance(x, str) else x
    )
    print(f"中文数据集加载完成，共 {len(chinese_df)} 条数据")
    
    # 读取英文数据集
    english_df = pd.read_csv(english_csv_path)
    english_df['content'] = english_df['content'].astype(str)
    print(f"英文数据集加载完成，共 {len(english_df)} 条数据")
    
    # 合并数据集
    combined_df = pd.concat([chinese_df, english_df], ignore_index=True)
    print(f"数据集合并完成，共 {len(combined_df)} 条数据")
    
    # 统计标签分布
    label_counts = combined_df['label'].value_counts()
    print("标签分布统计:")
    for label in emotion_labels:
        count = label_counts.get(label, 0)
        print(f"{label}: {count}")
    
    # 处理为多标签格式（当前是单标签，为了支持多标签，我们创建标签列表）
    combined_df['labels'] = combined_df['label'].apply(lambda x: [x] if isinstance(x, str) else [])
    
    # 过滤掉无效标签
    combined_df = combined_df[combined_df['labels'].apply(
        lambda x: all(l in emotion_labels for l in x)
    )]
    print(f"过滤后有效数据: {len(combined_df)} 条")
    
    return combined_df

# 自定义数据集类
class EmotionDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        labels = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        # 转换标签为多标签二值化格式
        label_tensor = torch.zeros(len(emotion_labels), dtype=torch.float)
        for label in labels:
            if label in emotion_labels:
                label_tensor[emotion_labels.index(label)] = 1.0
        
        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': label_tensor
        }

# 创建数据加载器
def create_data_loaders(df, tokenizer, batch_size=32, max_len=128):
    # 拆分训练集和验证集
    train_df, val_df = train_test_split(df, test_size=0.1, random_state=42)
    
    # 创建数据集实例
    train_dataset = EmotionDataset(
        texts=train_df['content'].tolist(),
        labels=train_df['labels'].tolist(),
        tokenizer=tokenizer,
        max_len=max_len
    )
    
    val_dataset = EmotionDataset(
        texts=val_df['content'].tolist(),
        labels=val_df['labels'].tolist(),
        tokenizer=tokenizer,
        max_len=max_len
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )
    
    print(f"训练集大小: {len(train_df)}, 验证集大小: {len(val_df)}")
    
    return train_loader, val_loader

# 初始化模型
def initialize_model():
    print("正在初始化模型...")
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-multilingual-cased')
    
    # 对于多标签分类，我们需要将模型修改为输出6个二分类任务
    model = DistilBertForSequenceClassification.from_pretrained(
        'distilbert-base-multilingual-cased',
        num_labels=len(emotion_labels),
        problem_type="multi_label_classification"
    )
    
    # 移动到GPU（如果可用）
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    print(f"模型初始化完成，使用设备: {device}")
    
    return tokenizer, model, device

# 训练函数
def train_model(model, tokenizer, train_loader, val_loader, device, epochs=4, learning_rate=2e-5):
    # 设置优化器
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    
    # 计算总步数
    total_steps = len(train_loader) * epochs
    
    # 设置学习率调度器
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )
    
    # 损失函数（二元交叉熵，适用于多标签分类）
    loss_fn = torch.nn.BCEWithLogitsLoss()
    
    # 存储最佳验证损失
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        print(f"\n===== 开始训练第 {epoch+1}/{epochs} 轮 =====")
        
        # 训练模式
        model.train()
        train_loss = 0
        
        for batch in tqdm(train_loader, desc=f"训练轮次 {epoch+1}"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # 梯度清零
            model.zero_grad()
            
            # 前向传播
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            
            # 计算损失
            loss = loss_fn(logits, labels)
            train_loss += loss.item()
            
            # 反向传播和优化
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
        
        # 计算平均训练损失
        avg_train_loss = train_loss / len(train_loader)
        print(f"训练损失: {avg_train_loss:.4f}")
        
        # 验证模式
        model.eval()
        val_loss = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="验证中"):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = model(input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                
                loss = loss_fn(logits, labels)
                val_loss += loss.item()
        
        # 计算平均验证损失
        avg_val_loss = val_loss / len(val_loader)
        print(f"验证损失: {avg_val_loss:.4f}")
        
        # 保存最佳模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            print(f"验证损失改善，保存模型...")
            save_model(model, tokenizer, "best_emotion_model")
    
    print(f"\n训练完成！最佳验证损失: {best_val_loss:.4f}")

# 保存模型
def save_model(model, tokenizer, model_dir):
    # 创建保存目录
    os.makedirs(model_dir, exist_ok=True)
    
    # 保存模型
    model.save_pretrained(model_dir)
    
    # 保存分词器
    tokenizer.save_pretrained(model_dir)
    
    # 保存标签映射
    with open(os.path.join(model_dir, 'labels.json'), 'w', encoding='utf-8') as f:
        json.dump(emotion_labels, f, ensure_ascii=False)
    
    print(f"模型已保存到 {model_dir}")

# 主函数
def main():
    # 数据集路径 - 使用原始字符串避免路径解析问题
    chinese_csv_path = r"d:\code\501\Emotion8\database\ChineseT.CSV"
    english_csv_path = r"d:\code\501\Emotion8\database\EnglishT.CSV"
    
    # 加载和预处理数据
    df = load_and_preprocess_data(chinese_csv_path, english_csv_path)
    
    # 初始化模型和分词器
    tokenizer, model, device = initialize_model()
    
    # 创建数据加载器
    train_loader, val_loader = create_data_loaders(df, tokenizer, batch_size=16, max_len=128)
    
    # 训练模型
    train_model(model, tokenizer, train_loader, val_loader, device, epochs=4, learning_rate=2e-5)
    
    # 保存最终模型
    save_model(model, tokenizer, "final_emotion_model")

if __name__ == "__main__":
    main()