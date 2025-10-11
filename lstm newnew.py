import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, losses, metrics
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from typing import Dict, List, Tuple, Any


# ------------------------------
# 1. 配置参数（适配多样本训练场景）
# ------------------------------
class Config:
    data_dir = r"/Users/rcchl/Desktop/ticket_data"  # 数据目录（train.csv所在路径）
    model_save_path = os.path.join(data_dir, "lstm_text_classifier.h5")  # 模型保存路径
    test_case_json_path = os.path.join(data_dir, "test_cases.json")  # 测试用例路径
    # 文本参数
    max_text_len = 30  # 词汇序列最大长度（根据样本调整）
    embedding_dim = 128  # 词嵌入维度
    lstm_units = 128  # LSTM单元数
    # 训练参数
    batch_size = 8  # 批次大小
    epochs = 25  # 训练轮次
    val_split = 0.2  # 验证集比例
    learning_rate = 0.01  # 学习率
    confidence_threshold = 0.7  # 分类置信度阈值


config = Config()


# ------------------------------
# 2. 数据加载与预处理（核心：从train.csv提取样本）
# ------------------------------
def load_and_preprocess_data(data_dir: str) -> Tuple[np.ndarray, np.ndarray, LabelEncoder, Tokenizer]:
    """加载train.csv并预处理为模型可接受的格式"""
    # 加载CSV文件
    df = pd.read_csv(os.path.join(data_dir, "train.csv"))
    # 提取特征列（按英文逗号分割为词汇列表）
    features = df["features"].apply(lambda x: x.split(",")).tolist()
    # 提取标签列
    labels = df["label"].tolist()

    # 标签编码（文本→数字索引）
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)
    num_classes = len(label_encoder.classes_)
    print(f"加载完成：{len(features)}个样本，{num_classes}个类别")
    print(f"类别列表：{label_encoder.classes_}")

    # 文本序列编码（词汇→整数索引）
    tokenizer = Tokenizer(filters="", oov_token="<OOV>")  # 保留所有词汇，未知词用<OOV>
    tokenizer.fit_on_texts(features)  # 基于训练样本构建词汇表
    vocab_size = len(tokenizer.word_index) + 1  # 词汇表大小（+1预留0值）
    print(f"词汇表大小：{vocab_size}")

    # 序列转换与长度统一（padding）
    sequences = tokenizer.texts_to_sequences(features)
    padded_sequences = pad_sequences(
        sequences, maxlen=config.max_text_len, padding="post", truncating="post"
    )

    return padded_sequences, encoded_labels, label_encoder, tokenizer


# ------------------------------
# 3. LSTM模型构建（具备自学习能力的分类模型）
# ------------------------------
def build_lstm_classifier(vocab_size: int, num_classes: int) -> models.Model:
    """构建用于文本分类的LSTM模型（含自学习能力）"""
    # 输入层（词汇序列）
    inputs = layers.Input(shape=(config.max_text_len,), name="text_input")

    # 词嵌入层（将整数索引转为稠密向量，学习词汇语义）
    x = layers.Embedding(
        input_dim=vocab_size,
        output_dim=config.embedding_dim,
        mask_zero=True,  # 忽略padding的0值
        name="embedding"
    )(inputs)

    # LSTM层（提取序列特征，学习词汇间的顺序关系）
    x = layers.LSTM(
        units=config.lstm_units,
        return_sequences=False,  # 仅输出最后一个时间步特征
        name="lstm"
    )(x)
    x = layers.Dropout(0.3)(x)  # 防止过拟合

    # 分类输出层（输出每个类别的概率）
    outputs = layers.Dense(
        units=num_classes,
        activation="softmax",  # 多分类用softmax
        name="output"
    )(x)

    # 定义模型
    model = models.Model(inputs=inputs, outputs=outputs)
    # 编译模型（配置优化器、损失函数、评估指标）
    model.compile(
        optimizer=optimizers.Adam(learning_rate=config.learning_rate),
        loss=losses.SparseCategoricalCrossentropy(),  # 标签为整数时用这个损失
        metrics=[metrics.SparseCategoricalAccuracy(name="accuracy")]
    )
    return model


# ------------------------------
# 4. 模型训练（核心：自学习过程）
# ------------------------------
def train_model(model: models.Model, x: np.ndarray, y: np.ndarray) -> models.Model:
    """训练模型，通过样本迭代优化参数（自学习核心）"""
    # 划分训练集和验证集（用验证集监控过拟合）
    x_train, x_val, y_train, y_val = train_test_split(
        x, y, test_size=config.val_split, random_state=42, stratify=y
    )

    # 训练模型（迭代更新参数）
    print("\n开始模型训练...")
    history = model.fit(
        x=x_train,
        y=y_train,
        batch_size=config.batch_size,
        epochs=config.epochs,
        validation_data=(x_val, y_val),
        verbose=1
    )

    # 保存训练好的模型
    model.save(config.model_save_path)
    print(f"\n模型训练完成，已保存至：{config.model_save_path}")

    # 打印最终准确率
    final_train_acc = history.history["accuracy"][-1]
    final_val_acc = history.history["val_accuracy"][-1]
    print(f"最终训练准确率：{final_train_acc:.4f}")
    print(f"最终验证准确率：{final_val_acc:.4f}")
    return model


# ------------------------------
# 5. 模型预测（基于训练好的参数）
# ------------------------------
def preprocess_input(text: List[str], tokenizer: Tokenizer) -> np.ndarray:
    """将输入词汇列表预处理为模型可接受的序列"""
    # 转换为整数序列
    sequence = tokenizer.texts_to_sequences([text])
    # 统一长度
    padded = pad_sequences(
        sequence, maxlen=config.max_text_len, padding="post", truncating="post"
    )
    return padded


def predict_label(model: models.Model, input_words: List[str], tokenizer: Tokenizer,
                  label_encoder: LabelEncoder) -> Dict[str, Any]:
    """使用训练好的模型预测输入词汇的类别"""
    # 预处理输入
    input_seq = preprocess_input(input_words, tokenizer)
    # 模型预测（输出每个类别的概率）
    pred_probs = model.predict(input_seq, verbose=0)[0]
    # 找到概率最高的类别
    max_prob_idx = np.argmax(pred_probs)
    pred_label = label_encoder.inverse_transform([max_prob_idx])[0]
    confidence = float(pred_probs[max_prob_idx])  # 置信度=最高概率

    # 结果判断
    if confidence >= config.confidence_threshold:
        return {
            "预测类别": pred_label,
            "置信度": round(confidence, 3),
            "状态": "分类成功"
        }
    else:
        return {
            "预测类别": "未知类别",
            "置信度": round(confidence, 3),
            "状态": f"置信度低于阈值（{config.confidence_threshold}）"
        }


# ------------------------------
# 6. 测试入口（验证模型效果）
# ------------------------------
def load_test_cases(json_path: str) -> List[Dict[str, Any]]:
    """加载测试用例（格式：每个用例包含名称和输入词汇列表）"""
    if not os.path.exists(json_path):
        print(f"测试用例文件不存在：{json_path}")
        return []
    with open(json_path, "r", encoding="utf-8") as f:
        test_data = json.load(f)
    return test_data.get("test_cases", [])


def run_test(model: models.Model, test_cases: List[Dict[str, Any]],
             tokenizer: Tokenizer, label_encoder: LabelEncoder):
    """用测试用例验证模型效果"""
    if not test_cases:
        print("\n无测试用例可执行")
        return
    print("\n" + "=" * 50)
    print("开始模型测试")
    print("=" * 50)
    for idx, case in enumerate(test_cases, 1):
        case_name = case.get("case_name", f"测试用例{idx}")
        input_words = case.get("input_words", [])
        print(f"\n【{case_name}】")
        print(f"输入词汇：{input_words}")
        try:
            result = predict_label(model, input_words, tokenizer, label_encoder)
            for k, v in result.items():
                print(f"  {k}：{v}")
        except Exception as e:
            print(f"  预测失败：{str(e)}")
    print("\n" + "=" * 50)
    print("测试完成")


# ------------------------------
# 主程序（全流程执行）
# ------------------------------
if __name__ == "__main__":
    # 1. 加载并预处理数据
    x, y, label_encoder, tokenizer = load_and_preprocess_data(config.data_dir)
    num_classes = len(label_encoder.classes_)

    # 2. 构建模型
    lstm_model = build_lstm_classifier(
        vocab_size=len(tokenizer.word_index) + 1,
        num_classes=num_classes
    )
    print("\n模型结构：")
    lstm_model.summary()

    # 3. 训练模型（自学习核心步骤）
    trained_model = train_model(lstm_model, x, y)

    # 4. 加载测试用例并执行预测
    test_cases = load_test_cases(config.test_case_json_path)
    run_test(trained_model, test_cases, tokenizer, label_encoder)