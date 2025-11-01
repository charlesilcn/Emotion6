import os
import time
import json
import numpy as np

# 禁用Hugging Face自动连接
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

class EmotionClassifier:
    def __init__(self, model_dir="best_emotion_model"):
        # 定义默认的情绪标签映射
        self.default_labels = ['happy', 'sad', 'angry', 'fear', 'surprise', 'neutral']
        
        # 英文到中文的情绪映射
        self.english_to_chinese = {
            'happy': '喜悦',
            'sad': '悲伤',
            'angry': '愤怒',
            'fear': '恐惧',
            'surprise': '惊讶',
            'neutral': '中性'
        }
        
        # 初始化变量
        self.tokenizer = None
        self.model = None
        self.labels = self.default_labels
        self.device = None
        self.use_mock = False
        
        try:
            # 尝试加载模型和分词器
            self.tokenizer = DistilBertTokenizer.from_pretrained(model_dir)
            self.model = DistilBertForSequenceClassification.from_pretrained(model_dir)
            
            # 尝试加载标签映射
            try:
                with open(os.path.join(model_dir, 'labels.json'), 'r', encoding='utf-8') as f:
                    self.labels = json.load(f)
            except Exception:
                print("警告: 标签文件加载失败，使用默认标签")
                self.labels = self.default_labels
            
            # 移动到GPU（如果可用）
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model.to(self.device)
            self.model.eval()
            
            print(f"模型加载完成，使用设备: {self.device}")
            print(f"支持的情绪标签: {self.labels}")
        except Exception as e:
            print(f"警告: 无法加载模型，将使用模拟模式: {e}")
            self.use_mock = True
            self.labels = self.default_labels
            self.device = torch.device('cpu')
    
    def _mock_predict(self, text, threshold=0.5, lang='auto'):
        """模拟预测功能，用于演示"""
        # 简单的情绪关键词匹配规则
        keywords = {
            'happy': ['开心', '快乐', '高兴', '愉悦', '幸福', '满足', '兴奋', '棒', '好', 'nice', 'happy', 'excited', 'great', 'good'],
            'sad': ['难过', '伤心', '悲伤', '忧郁', '沮丧', '不幸', '痛苦', '难受', 'sad', 'upset', 'depressed', 'unhappy'],
            'angry': ['生气', '愤怒', '生气', '恼火', '烦', '怒', '气死', '不爽', 'angry', 'mad', 'upset', 'frustrated'],
            'fear': ['害怕', '恐惧', '惊恐', '吓死', '担心', '恐怖', '畏惧', '害怕', 'fear', 'scared', 'terrified', 'afraid'],
            'surprise': ['惊讶', '惊喜', '震惊', '没想到', '居然', '哇', 'amazing', 'surprised', 'shocked', 'unbelievable'],
            'neutral': ['一般', '普通', '正常', '平常', '还行', 'okay', 'normal', 'regular', 'average']
        }
        
        # 计算每个情绪的匹配分数
        scores = {}
        text_lower = text.lower()
        
        for emotion, words in keywords.items():
            count = sum(1 for word in words if word.lower() in text_lower)
            # 基础分数 + 关键词匹配分数
            scores[emotion] = 0.1 + (count * 0.2) if count > 0 else 0.1
        
        # 归一化分数
        total = sum(scores.values())
        if total > 0:
            scores = {k: v/total for k, v in scores.items()}
        
        # 根据阈值确定预测的情绪
        predicted_emotions = []
        emotion_scores = {}
        
        for emotion in self.labels:
            score = scores.get(emotion, 0.1)
            emotion_scores[emotion] = float(score)
            if score >= threshold:
                predicted_emotions.append(emotion)
        
        # 如果没有预测到任何情绪，返回概率最高的一个
        if not predicted_emotions:
            max_emotion = max(scores.items(), key=lambda x: x[1])[0]
            predicted_emotions.append(max_emotion)
        
        # 确定输出语言
        if lang == 'auto':
            # 简单检测是否包含中文字符
            has_chinese = any('\u4e00' <= char <= '\u9fff' for char in text)
            output_lang = 'zh' if has_chinese else 'en'
        else:
            output_lang = lang
        
        # 转换标签语言
        if output_lang == 'zh':
            predicted_emotions = [self.english_to_chinese.get(emotion, emotion) for emotion in predicted_emotions]
            emotion_scores = {
                self.english_to_chinese.get(label, label): score 
                for label, score in emotion_scores.items()
            }
        
        return {
            'text': text,
            'predicted_emotions': predicted_emotions,
            'emotion_scores': emotion_scores,
            'language': output_lang,
            'processing_time': "15.23ms"  # 模拟时间
        }
    
    def predict(self, text, threshold=0.5, lang='auto'):
        """
        预测文本的情绪
        
        参数:
        text: 输入文本
        threshold: 置信度阈值，用于多标签分类
        lang: 输出语言，'en'为英文，'zh'为中文，'auto'自动检测
        
        返回:
        字典，包含预测结果和置信度
        """
        start_time = time.time()
        
        # 如果使用模拟模式，调用模拟预测
        if self.use_mock or not self.model or not self.tokenizer:
            print(f"使用模拟模式预测文本情绪: {text[:50]}...")
            return self._mock_predict(text, threshold, lang)
        
        try:
            # 标记化输入
            encoding = self.tokenizer(
                text,
                add_special_tokens=True,
                max_length=128,
                return_token_type_ids=False,
                padding='max_length',
                truncation=True,
                return_attention_mask=True,
                return_tensors='pt'
            )
            
            # 移动到设备
            input_ids = encoding['input_ids'].to(self.device)
            attention_mask = encoding['attention_mask'].to(self.device)
            
            # 推理
            with torch.no_grad():
                outputs = self.model(input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                # 使用sigmoid激活函数获取概率
                probabilities = torch.sigmoid(logits).cpu().numpy()[0]
            
            # 根据阈值确定预测的情绪
            predicted_emotions = []
            emotion_scores = {}
            
            for i, label in enumerate(self.labels):
                score = probabilities[i]
                emotion_scores[label] = float(score)
                if score >= threshold:
                    predicted_emotions.append(label)
            
            # 如果没有预测到任何情绪，返回概率最高的一个
            if not predicted_emotions:
                max_label = self.labels[np.argmax(probabilities)]
                predicted_emotions.append(max_label)
            
            # 确定输出语言
            if lang == 'auto':
                # 简单检测是否包含中文字符
                has_chinese = any('\u4e00' <= char <= '\u9fff' for char in text)
                output_lang = 'zh' if has_chinese else 'en'
            else:
                output_lang = lang
            
            # 转换标签语言
            if output_lang == 'zh':
                predicted_emotions = [self.english_to_chinese.get(emotion, emotion) for emotion in predicted_emotions]
                emotion_scores = {
                    self.english_to_chinese.get(label, label): score 
                    for label, score in emotion_scores.items()
                }
            
            processing_time = round((time.time() - start_time) * 1000, 2)  # 毫秒
            
            return {
                'text': text,
                'predicted_emotions': predicted_emotions,
                'emotion_scores': emotion_scores,
                'language': output_lang,
                'processing_time': f"{processing_time}ms"
            }
        except Exception as e:
            print(f"预测过程出错，使用模拟模式: {e}")
            # 出错时回退到模拟模式
            return self._mock_predict(text, threshold, lang)
    
    def batch_predict(self, texts, threshold=0.5, lang='auto'):
        """批量预测多个文本"""
        start_time = time.time()
        results = []
        
        try:
            # 逐个处理文本
            for text in texts:
                result = self.predict(text, threshold, lang)
                results.append(result)
            
            processing_time = round((time.time() - start_time) * 1000, 2)  # 毫秒
            
            return {
                'results': results,
                'total': len(texts),
                'processing_time': f"{processing_time}ms"
            }
        except Exception as e:
            print(f"批量预测出错: {e}")
            # 出错时返回基本结果格式
            return {
                'results': results,
                'total': len(texts),
                'processing_time': "0ms",
                'error': str(e)
            }

# 测试函数
def test_classifier():
    print("初始化情绪分类器...")
    classifier = EmotionClassifier()
    
    print("\n开始测试...")
    
    # 测试中文文本
    chinese_tests = [
        "今天天气真好，心情特别愉快！",
        "又惊又喜，没想到会收到这么好的礼物！",
        "这个消息太可怕了，我感到非常恐惧和担忧。",
        "我很生气，为什么总是这样对我？",
        "他看起来很伤心，似乎遇到了什么困难。"
    ]
    
    print("\n中文测试结果:")
    for text in chinese_tests:
        result = classifier.predict(text)
        print(f"文本: {text}")
        print(f"预测情绪: {result['predicted_emotions']}")
        print(f"情绪得分: {result['emotion_scores']}")
        print("---")
    
    # 测试英文文本
    english_tests = [
        "I am so happy to see you again!",
        "I was surprised and delighted by the news.",
        "This situation makes me feel angry and frustrated.",
        "I am feeling sad and lonely today.",
        "The movie was really scary, I was filled with fear."
    ]
    
    print("\n英文测试结果:")
    for text in english_tests:
        result = classifier.predict(text)
        print(f"Text: {text}")
        print(f"Predicted emotions: {result['predicted_emotions']}")
        print(f"Emotion scores: {result['emotion_scores']}")
        print("---")

if __name__ == "__main__":
    # 需要numpy库
    import numpy as np
    
    # 检查模型是否存在
    if os.path.exists("best_emotion_model"):
        test_classifier()
    else:
        print("警告：未找到训练好的模型。请先运行train_emotion_model.py进行训练。")
        print("\n如果需要直接测试，可以使用以下代码初始化分类器:")
        print("""
        from infer_emotion_model import EmotionClassifier
        classifier = EmotionClassifier("path_to_your_model")  # 替换为实际模型路径
        result = classifier.predict("你的测试文本")
        print(result)
        """)