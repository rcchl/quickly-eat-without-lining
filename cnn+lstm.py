import os
import re
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, losses
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from PIL import Image
from typing import Dict, List, Tuple, Any


# ------------------------------
# 1. 配置参数（新增动态提示词路径，整合所有外部文件路径）
# ------------------------------
class Config:
    data_dir = "/Users/rcchl/Desktop/ticket_data"  # 数据根目录（需确认是否匹配你的实际路径）
    # 外部文件路径（全部放在data_dir下，统一管理）
    rule_json_path = os.path.join(data_dir, "ticket_rules.json")  # 票据规则文件
    prompt_json_path = os.path.join(data_dir, "dynamic_prompts.json")  # 动态提示词（关键词）文件
    test_case_json_path = os.path.join(data_dir, "test_cases.json")  # 测试用例文件
    # ResNet50权重路径（需替换为你的本地权重路径）
    resnet_weights_path = "/Users/rcchl/PycharmProjects/PythonProject2/weights/resnet50_weights.h5"

    # 模型与训练参数
    image_size = (224, 224)
    max_text_len = 50  # LSTM文本序列长度（适配纯字段名）
    embedding_dim = 64  # 文本嵌入维度
    lstm_units = 64  # LSTM单元数
    # 少样本元学习参数
    num_ways = 2
    num_support = 1
    num_query = 1
    meta_batch_size = 2
    inner_lr = 0.001
    outer_lr = 0.0005
    num_inner_steps = 1
    num_epochs = 8
    confidence_threshold = 0.7  # 分类置信度阈值


config = Config()


# ------------------------------
# 2. 加载外部配置文件（规则+动态提示词，均内部调用）
# ------------------------------
def load_ticket_rules_from_json(json_path: str) -> Dict[str, Any]:
    """从JSON加载票据分类规则，重建format_check/logic_check逻辑"""
    with open(json_path, "r", encoding="utf-8") as f:
        raw_rules = json.load(f)

    ticket_rules = {}
    for ticket_type, raw_rule in raw_rules.items():
        # 重建format_check：基于JSON中的条件描述
        fmt_conditions = raw_rule["format_check_conditions"]

        def build_format_check(conditions):
            def format_check(ocr: str, struct: Dict[str, Any]) -> bool:
                for cond in conditions:
                    if cond["type"] == "in_ocr":
                        if cond["value"] not in ocr:
                            return False
                    elif cond["type"] == "in_struct_image_features":
                        if cond["value"] not in struct.get("image_features", []):
                            return False
                    elif cond["type"] == "struct_layout_eq":
                        if struct.get("layout") != cond["value"]:
                            return False
                    elif cond["type"] == "struct_shape_eq":
                        if struct.get("shape") != cond["value"]:
                            return False
                    elif cond["type"] == "struct_background_color_eq":
                        if struct.get("background_color") != cond["value"]:
                            return False
                return True

            return format_check

        # 重建logic_check：基于最小字段匹配数量
        logic_cond = raw_rule["logic_check_conditions"]

        def build_logic_check(min_count: int, field_list: List[str]):
            def logic_check(struct: Dict[str, Any]) -> bool:
                existing_fields = struct.get("key_fields_exist", [])
                matched_count = sum(1 for f in field_list if f in existing_fields)
                return matched_count >= min_count

            return logic_check

        # 组装完整规则
        ticket_rules[ticket_type] = {
            "core_name": raw_rule["core_name"],
            "key_fields": raw_rule["key_fields"],
            "format_check": build_format_check(fmt_conditions),
            "logic_check": build_logic_check(
                min_count=logic_cond["min_field_count"],
                field_list=logic_cond["field_list"]
            )
        }
    return ticket_rules


def load_dynamic_prompts(json_path: str) -> Dict[str, List[str]]:
    """加载动态提示词（关键词列表），仅用于内部分类增强，不输出"""
    with open(json_path, "r", encoding="utf-8") as f:
        prompts = json.load(f)
    # 确保包含所有票据类型+未知票据（避免KeyError）
    required_types = list(ticket_rules.keys()) + ["未知票据"]
    for t in required_types:
        if t not in prompts:
            prompts[t] = []
    return prompts


# 加载外部配置（顺序：先加载规则，再加载提示词）
ticket_rules = load_ticket_rules_from_json(config.rule_json_path)
dynamic_prompts = load_dynamic_prompts(config.prompt_json_path)


# ------------------------------
# 3. 规则匹配函数（核心改进：结合动态提示词增强分类判断）
# ------------------------------
def rule_based_classify(ocr_text: str, structure: Dict[str, Any]) -> Tuple[str, float]:
    max_score = 0.0
    result_type = "未知票据"

    all_possible_fields = list(set(sum([r["key_fields"] for r in ticket_rules.values()], [])))
    existing_fields = [field for field in all_possible_fields if field in ocr_text]
    structure["key_fields_exist"] = existing_fields

    for ticket_type, rules in ticket_rules.items():
        score = 0.0
        type_prompts = dynamic_prompts.get(ticket_type, [])

        # 1. 核心名称匹配（固定0.3，不超额）
        core_match = any(name in ocr_text for name in rules["core_name"])
        if not core_match:
            continue
        score += 0.3

        # 2. 关键字段匹配（最高0.4，含提示词加分）
        base_field_matched = sum(1 for f in rules["key_fields"] if f in existing_fields)
        prompt_field_matched = sum(1 for f in existing_fields if f in type_prompts)
        # 限制该部分总分≤0.4（提示词加分最多0.2）
        total_field_score = (base_field_matched + min(prompt_field_matched, 2) * 0.5) / len(rules["key_fields"])
        field_contribution = min(total_field_score * 0.4, 0.4)  # 上限0.4
        score += field_contribution

        # 3. 格式特征匹配（最高0.3，含视觉提示词加分）
        base_format_ok = rules["format_check"](ocr_text, structure)
        if base_format_ok:
            base_format_score = 0.3
            # 视觉提示词最多加0.1（即最多匹配2个，每个0.05）
            struct_visual_feats = structure.get("image_features", []) + [
                structure.get("layout", ""), structure.get("background_color", ""), structure.get("shape", "")
            ]
            visual_prompt_matched = sum(1 for feat in struct_visual_feats if feat in type_prompts)
            visual_add = min(visual_prompt_matched * 0.05, 0.1)  # 上限0.1
            format_contribution = min(base_format_score + visual_add, 0.3)  # 总上限0.3
            score += format_contribution

        # 4. 逻辑校验
        if not rules["logic_check"](structure):
            continue

        # 最终限制总分≤1.0（防止任何情况下溢出）
        score = min(score, 1.0)

        if score > max_score and score >= config.confidence_threshold:
            max_score = score
            result_type = ticket_type

    return (result_type, round(max_score, 3))  # 保留3位小数


# ------------------------------
# 4. 数据加载与预处理（适配LSTM输入，无修改）
# ------------------------------
def load_training_data(data_dir: str):
    """加载训练数据（train.csv），返回预处理后的图像路径、文本、结构、标签"""
    df = pd.read_csv(os.path.join(data_dir, "train.csv"))
    # 图像绝对路径
    image_paths = [os.path.join(data_dir, p) for p in df["image_path"].values]
    # OCR文本（纯字段名）
    ocr_texts = df["ocr_text"].values.tolist()
    # 结构信息（视觉特征，eval转字典）
    structures = df["structure"].apply(eval).tolist()
    # 标签编码
    labels = df["label"].values.tolist()
    le = LabelEncoder()
    encoded_labels = le.fit_transform(labels)

    return image_paths, ocr_texts, structures, encoded_labels, le


def preprocess_image(img_path: str) -> tf.Tensor:
    """图像预处理：Resize→转数组→ResNet50预处理"""
    img = Image.open(img_path).convert("RGB")
    img = img.resize(config.image_size)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    return tf.keras.applications.resnet50.preprocess_input(img_array)


def create_text_tokenizer(texts: List[str]) -> Tokenizer:
    """创建文本Tokenizer（按空格分割纯字段名，适配LSTM）"""

    def tokenize(text):
        return text.strip().split()

    tokenized_texts = [tokenize(t) for t in texts]
    tokenizer = Tokenizer(filters="", oov_token="<OOV>")  # 保留所有字符，OOV处理未见过的字段名
    tokenizer.fit_on_texts(tokenized_texts)
    return tokenizer


def create_meta_task(image_paths: List[str], ocr_texts: List[str],
                     encoded_labels: np.ndarray, tokenizer: Tokenizer) -> Tuple[Tuple, Tuple]:
    """生成少样本元学习任务（支持集+查询集）"""
    unique_labels = np.unique(encoded_labels)
    # 采样类别（不足时重复采样）
    sampled_labels = np.random.choice(
        unique_labels, size=config.num_ways, replace=len(unique_labels) < config.num_ways
    )

    support_imgs, support_texts, support_labels = [], [], []
    query_imgs, query_texts, query_labels = [], [], []

    for label in sampled_labels:
        label_idxs = np.where(encoded_labels == label)[0]
        required = config.num_support + config.num_query
        # 样本不足时重复采样
        selected_idxs = np.random.choice(
            label_idxs, size=required, replace=len(label_idxs) < required
        )

        # 分割支持集与查询集
        support_idxs = selected_idxs[:config.num_support]
        query_idxs = selected_idxs[config.num_support:]

        # 处理支持集
        for idx in support_idxs:
            support_imgs.append(preprocess_image(image_paths[idx]))
            # 文本转LSTM序列
            tokenized = ocr_texts[idx].strip().split()
            seq = tokenizer.texts_to_sequences([tokenized])[0]
            support_texts.append(pad_sequences([seq], maxlen=config.max_text_len)[0])
            support_labels.append(label)

        # 处理查询集
        for idx in query_idxs:
            query_imgs.append(preprocess_image(image_paths[idx]))
            tokenized = ocr_texts[idx].strip().split()
            seq = tokenizer.texts_to_sequences([tokenized])[0]
            query_texts.append(pad_sequences([seq], maxlen=config.max_text_len)[0])
            query_labels.append(label)

    # 转为Tensor格式
    support_tuple = (
        tf.convert_to_tensor(support_imgs, dtype=tf.float32),
        tf.convert_to_tensor(support_texts, dtype=tf.int32),
        tf.convert_to_tensor(support_labels, dtype=tf.int32)
    )
    query_tuple = (
        tf.convert_to_tensor(query_imgs, dtype=tf.float32),
        tf.convert_to_tensor(query_texts, dtype=tf.int32),
        tf.convert_to_tensor(query_labels, dtype=tf.int32)
    )
    return support_tuple, query_tuple


def create_meta_batch(image_paths: List[str], ocr_texts: List[str],
                      encoded_labels: np.ndarray, tokenizer: Tokenizer) -> List[Tuple[Tuple, Tuple]]:
    """生成元学习批量任务"""
    return [create_meta_task(image_paths, ocr_texts, encoded_labels, tokenizer)
            for _ in range(config.meta_batch_size)]


# 加载训练数据与Tokenizer
image_paths, ocr_texts, structures, encoded_labels, label_encoder = load_training_data(config.data_dir)
text_tokenizer = create_text_tokenizer(ocr_texts)
vocab_size = len(text_tokenizer.word_index) + 1  # 词汇表大小（+1用于0填充）


# ------------------------------
# 5. 多模态原型网络（文本分支为LSTM，图像分支为ResNet50）
# ------------------------------
def build_lstm_multimodal_model(vocab_size: int) -> models.Model:
    """构建LSTM文本+ResNet50图像的多模态原型网络"""
    # 1. 图像分支（ResNet50提取视觉特征）
    img_input = layers.Input(shape=(*config.image_size, 3), name="image_input")
    resnet_base = ResNet50(
        weights=config.resnet_weights_path,
        include_top=False,
        input_tensor=img_input
    )
    # 解冻部分层学习票据视觉特征
    for layer in resnet_base.layers[:-10]:
        layer.trainable = False
    img_feat = resnet_base.output
    img_feat = layers.GlobalAveragePooling2D()(img_feat)
    img_feat = layers.Dense(256, activation="relu")(img_feat)
    img_feat = layers.BatchNormalization()(img_feat)

    # 2. 文本分支（LSTM提取序列特征）
    text_input = layers.Input(shape=(config.max_text_len,), name="text_input")
    text_feat = layers.Embedding(
        input_dim=vocab_size,
        output_dim=config.embedding_dim,
        input_length=config.max_text_len,
        mask_zero=True  # 忽略0填充对LSTM的影响
    )(text_input)
    text_feat = layers.LSTM(
        units=config.lstm_units,
        return_sequences=False  # 仅输出最后一步特征（适配分类）
    )(text_feat)
    text_feat = layers.Dense(128, activation="relu")(text_feat)

    # 3. 特征融合（视觉+文本）
    fused_feat = layers.Concatenate()([img_feat, text_feat])
    fused_feat = layers.Dense(256, activation="relu")(fused_feat)
    fused_feat = layers.Dropout(0.2)(fused_feat)  # 防止过拟合

    return models.Model(inputs=[img_input, text_input], outputs=fused_feat)


def compute_class_prototypes(features: tf.Tensor, labels: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    """计算类别原型：同类特征的平均值"""
    unique_labels = tf.unique(labels)[0]
    prototypes = []
    for label in unique_labels:
        class_feats = tf.boolean_mask(features, tf.equal(labels, label))
        prototypes.append(tf.reduce_mean(class_feats, axis=0))
    return tf.stack(prototypes, axis=0), unique_labels


def compute_euclidean_distance(features: tf.Tensor, prototypes: tf.Tensor) -> tf.Tensor:
    """计算特征与原型的欧氏距离（加1e-6避免除以0）"""
    feat_expand = tf.expand_dims(features, axis=1)
    proto_expand = tf.expand_dims(prototypes, axis=0)
    return tf.sqrt(tf.reduce_sum(tf.square(feat_expand - proto_expand), axis=-1) + 1e-6)


# ------------------------------
# 6. 元学习训练（适配LSTM多模态模型）
# ------------------------------
def meta_train_step(model: models.Model, meta_task: Tuple[Tuple, Tuple],
                    inner_lr: float, num_inner_steps: int) -> Tuple[tf.Tensor, List[tf.Tensor]]:
    """元学习单步训练：内循环适应支持集，外循环更新元参数"""
    (support_imgs, support_texts, support_labels), (query_imgs, query_texts, query_labels) = meta_task

    # 保存初始权重（外循环后恢复）
    initial_weights = model.get_weights()
    # 内循环优化器（legacy.Adam适配M1/M2 Mac）
    inner_optimizer = optimizers.legacy.Adam(learning_rate=inner_lr)
    loss_fn = losses.SparseCategoricalCrossentropy(from_logits=True)

    # 内循环：适应支持集
    for _ in range(num_inner_steps):
        with tf.GradientTape() as tape:
            # 提取支持集特征
            support_feats = model([support_imgs, support_texts], training=True)
            # 计算类别原型与距离
            support_protos, unique_support_labels = compute_class_prototypes(support_feats, support_labels)
            support_distances = compute_euclidean_distance(support_feats, support_protos)
            support_logits = -support_distances  # 距离越小，logits越大

            # 标签映射（支持集标签→0~num_ways-1）
            label_mapping = tf.lookup.StaticHashTable(
                tf.lookup.KeyValueTensorInitializer(unique_support_labels,
                                                    tf.range(tf.shape(unique_support_labels)[0])),
                default_value=-1
            )
            mapped_support_labels = label_mapping.lookup(support_labels)
            # 计算损失（加1e-8避免数值异常）
            support_loss = loss_fn(mapped_support_labels, support_logits + 1e-8)

        # 梯度更新
        grads = tape.gradient(support_loss, model.trainable_variables)
        inner_optimizer.apply_gradients(zip(grads, model.trainable_variables))

    # 外循环：用查询集计算元损失
    with tf.GradientTape() as tape:
        query_feats = model([query_imgs, query_texts], training=True)
        query_distances = compute_euclidean_distance(query_feats, support_protos)
        query_logits = -query_distances
        mapped_query_labels = label_mapping.lookup(query_labels)
        query_loss = loss_fn(mapped_query_labels, query_logits + 1e-8)

    # 恢复初始权重（仅外循环更新元参数）
    model.set_weights(initial_weights)
    # 计算元梯度
    meta_grads = tape.gradient(query_loss, model.trainable_variables)
    return query_loss, meta_grads


def run_meta_training(model: models.Model, image_paths: List[str], ocr_texts: List[str],
                      encoded_labels: np.ndarray, tokenizer: Tokenizer):
    """执行元学习训练"""
    outer_optimizer = optimizers.legacy.Adam(learning_rate=config.outer_lr)
    print("开始元学习训练...")
    for epoch in range(config.num_epochs):
        total_meta_loss = 0.0
        # 生成元批量任务
        meta_batch = create_meta_batch(image_paths, ocr_texts, encoded_labels, tokenizer)
        # 遍历元批量任务更新
        for task in meta_batch:
            task_loss, task_grads = meta_train_step(
                model, task, config.inner_lr, config.num_inner_steps
            )
            outer_optimizer.apply_gradients(zip(task_grads, model.trainable_variables))
            total_meta_loss += task_loss.numpy()
        # 打印每轮损失
        avg_meta_loss = total_meta_loss / len(meta_batch)
        print(f"Epoch {epoch + 1:2d}/{config.num_epochs} | 平均元损失：{avg_meta_loss:.4f}")
    print("元学习训练完成！")


# 初始化模型并训练
multimodal_model = build_lstm_multimodal_model(vocab_size)
print("多模态模型结构：")
multimodal_model.summary()
run_meta_training(multimodal_model, image_paths, ocr_texts, encoded_labels, text_tokenizer)


# ------------------------------
# 7. 融合预测函数（模型+规则双保险，无提示词输出）
# ------------------------------
def predict_ticket_type(model: models.Model, img_path: str, ocr_text: str,
                        structure: Dict[str, Any], tokenizer: Tokenizer,
                        label_encoder: LabelEncoder) -> Dict[str, Any]:
    """预测票据类型：模型预测+规则匹配融合"""
    # 1. 预处理输入
    # 图像预处理
    img_tensor = preprocess_image(img_path)
    img_tensor = tf.expand_dims(img_tensor, axis=0)
    # 文本预处理（转LSTM序列）
    tokenized_text = ocr_text.strip().split()
    text_seq = tokenizer.texts_to_sequences([tokenized_text])[0]
    text_seq = pad_sequences([text_seq], maxlen=config.max_text_len)

    # 2. 模型预测（无梯度计算）
    model.trainable = False
    sample_feat = model([img_tensor, text_seq])
    # 生成临时原型（实际应用可缓存所有类别原型）
    temp_prototype, _ = compute_class_prototypes(sample_feat, tf.convert_to_tensor([0]))
    distance = compute_euclidean_distance(sample_feat, temp_prototype).numpy()[0][0]
    max_possible_distance = 10.0  # 根据实际数据分布调整（可设为训练集中最大距离）
    model_confidence = 1.0 - min(distance / max_possible_distance, 1.0)
    model_confidence = round(float(model_confidence), 3)
    # 模型预测类别（基于训练标签映射）
    model_pred_label_idx = 0  # 临时占位，实际应用需匹配所有类别原型
    model_pred_label = label_encoder.inverse_transform([model_pred_label_idx])[0]

    # 3. 规则匹配预测
    rule_pred_label, rule_confidence = rule_based_classify(ocr_text, structure)

    # 4. 融合结果（高置信度优先）
    # 在predict_ticket_type函数中，交换判断顺序
    if rule_confidence >= config.confidence_threshold:  # 规则优先
        return {
            "票据类型": rule_pred_label,
            "置信度": rule_confidence,
            "分类来源": "规则匹配（优先）"
        }
    elif model_confidence >= config.confidence_threshold:  # 模型后补
        return {
            "票据类型": model_pred_label,
            "置信度": round(float(model_confidence), 3),
            "分类来源": "LSTM多模态模型"
        }
    else:
        return {
            "票据类型": "未知票据",
            "置信度": 0.0,
            "分类来源": "字段名/视觉特征不足，模型与规则均未匹配"
        }


# ------------------------------
# 8. 测试入口（从外部JSON加载测试用例，无提示词输出）
# ------------------------------
def load_test_cases(json_path: str) -> List[Dict[str, Any]]:
    """从JSON加载测试用例"""
    with open(json_path, "r", encoding="utf-8") as f:
        test_data = json.load(f)
    return test_data.get("test_cases", [])


def run_ticket_test(model: models.Model, test_cases: List[Dict[str, Any]],
                    tokenizer: Tokenizer, label_encoder: LabelEncoder):
    """执行票据分类测试，仅输出分类结果"""
    print("\n" + "=" * 50)
    print("开始票据分类测试（仅输出分类结果，无提示词）")
    print("=" * 50)
    for idx, case in enumerate(test_cases, 1):
        print(f"\n【测试用例 {idx}】{case.get('case_name', '未命名用例')}")
        # 提取测试参数
        img_relative_path = case.get("img_relative_path", "")
        ocr_text = case.get("ocr_text", "")
        structure = case.get("structure", {})
        # 补全图像绝对路径
        img_abs_path = os.path.join(config.data_dir, img_relative_path)

        # 执行预测
        try:
            result = predict_ticket_type(
                model, img_abs_path, ocr_text, structure,
                tokenizer, label_encoder
            )
            # 打印结果（仅分类相关信息）
            for k, v in result.items():
                print(f"  {k}：{v}")
        except Exception as e:
            print(f"  测试失败：{str(e)}")
    print("\n" + "=" * 50)
    print("测试完成")
    print("=" * 50)


# 加载测试用例并执行测试
if __name__ == "__main__":
    # 加载外部测试用例（从test_cases.json）
    test_cases = load_test_cases(config.test_case_json_path)
    if not test_cases:
        print("未加载到测试用例，请检查test_cases.json文件！")
    else:
        run_ticket_test(multimodal_model, test_cases, text_tokenizer, label_encoder)