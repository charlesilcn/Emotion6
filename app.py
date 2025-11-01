import os
import json
import pandas as pd
from flask import Flask, request, jsonify, render_template, redirect, url_for, flash, send_from_directory
from infer_emotion_model import EmotionClassifier
import numpy as np

# 设置中文显示
def setup_globals():
    global emotion_classifier
    # 初始化情绪分类器
    emotion_classifier = EmotionClassifier(model_dir="best_emotion_model")
    print("情绪分类器初始化完成")

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'csv'}

# 创建上传文件夹
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze_text():
    try:
        data = request.get_json()
        text = data.get('text', '')
        threshold = float(data.get('threshold', 0.5))
        
        if not text:
            return jsonify({'error': 'Text cannot be empty'}), 400
        
        # 预测情绪
        result = emotion_classifier.predict(text, threshold=threshold, lang='en')
        
        # 确保emotion_scores是英文的
        english_scores = {}
        for key, value in result['emotion_scores'].items():
            # 转换回英文标签（如果是中文的话）
            if key in emotion_classifier.english_to_chinese.values():
                # 找到对应的英文键
                for en_key, zh_key in emotion_classifier.english_to_chinese.items():
                    if zh_key == key:
                        english_scores[en_key] = value
                        break
            else:
                english_scores[key] = value
        
        result['emotion_scores'] = english_scores
        
        # 确保predicted_emotions也是英文的
        english_emotions = []
        for emotion in result['predicted_emotions']:
            if emotion in emotion_classifier.english_to_chinese.values():
                # 找到对应的英文键
                for en_key, zh_key in emotion_classifier.english_to_chinese.items():
                    if zh_key == emotion:
                        english_emotions.append(en_key)
                        break
            else:
                english_emotions.append(emotion)
        
        result['predicted_emotions'] = english_emotions
        
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/batch_analyze', methods=['POST'])
def batch_analyze():
    try:
        # 检查是否有文件
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400
        
        file = request.files['file']
        threshold = float(request.form.get('threshold', 0.5))
        
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        
        if file and allowed_file(file.filename):
            # 保存文件
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)
            
            try:
                # 读取CSV文件
                df = pd.read_csv(filepath)
                
                # 检查是否有'text'列
                if 'text' not in df.columns:
                    # 尝试使用第一列
                    text_column = df.columns[0]
                    texts = df[text_column].tolist()
                else:
                    texts = df['text'].tolist()
                
                # 过滤空文本
                texts = [str(text).strip() for text in texts if pd.notna(text) and str(text).strip()]
                
                if not texts:
                    return jsonify({'error': 'No valid text found in the file'}), 400
                
                # 批量预测
                batch_results = emotion_classifier.batch_predict(texts, threshold=threshold, lang='en')
                
                # 确保所有结果都是英文的
                for result in batch_results['results']:
                    # 转换emotion_scores到英文
                    english_scores = {}
                    for key, value in result['emotion_scores'].items():
                        if key in emotion_classifier.english_to_chinese.values():
                            for en_key, zh_key in emotion_classifier.english_to_chinese.items():
                                if zh_key == key:
                                    english_scores[en_key] = value
                                    break
                        else:
                            english_scores[key] = value
                    result['emotion_scores'] = english_scores
                    
                    # 转换predicted_emotions到英文
                    english_emotions = []
                    for emotion in result['predicted_emotions']:
                        if emotion in emotion_classifier.english_to_chinese.values():
                            for en_key, zh_key in emotion_classifier.english_to_chinese.items():
                                if zh_key == emotion:
                                    english_emotions.append(en_key)
                                    break
                        else:
                            english_emotions.append(emotion)
                    result['predicted_emotions'] = english_emotions
                
                return jsonify(batch_results)
            finally:
                # 删除上传的文件
                if os.path.exists(filepath):
                    os.remove(filepath)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

    # 添加README文件访问路由
    @app.route('/<filename>')
    def serve_readme(filename):
        # 只允许访问README文件
        if filename not in ['README.md', 'READMECN.md']:
            return jsonify({'error': 'File not allowed'}), 403
        try:
            return send_from_directory(os.path.dirname(os.path.abspath(__file__)), filename)
        except Exception as e:
            return jsonify({'error': str(e)}), 404

if __name__ == '__main__':
    setup_globals()
    app.run(debug=True, host='0.0.0.0', port=5000)