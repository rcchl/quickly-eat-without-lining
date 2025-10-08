import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, losses
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from PIL import Image
from typing import Dict, List, Tuple, Any


# ------------------------------
# 1. 配置参数（固定ticket_rules路径）
# ------------------------------
class Config:
    data_dir = "/Users/rcchl/Desktop/ticket_data"  # 数据根目录
    # 外部文件路径（已按你的路径配置）
    rule_json_path = os.path.join(data_dir, "ticket_rules.json")  # 你的ticket_rules路径
    prompt_json_path = os.path.join(data_dir, "dynamic_prompts.json")
    test_case_json_path = os.path.join(data_dir, "test_cases.json")
    prototype_save_path = os.path.join(data_dir, "class_prototypes.npy")  # 特征值字典保存路径

    # 模型与训练参数
    image_size = (224, 224)
    cnn_output_dim = 256
    max_text_len = 50
    embedding_dim = 64
    lstm_units = 64
    # 元学习参数
    num_ways = 2
    num_support = 1
    num_query = 1
    meta_batch_size = 2
    inner_lr = 0.001
    outer_lr = 0.0005
    num_inner_steps = 1
    num_epochs = 8
    confidence_threshold = 0.7


config = Config()


# ------------------------------
# 2. 加载外部配置文件（适配你的ticket_rules）
# ------------------------------
def load_ticket_rules_from_json(json_path: str) -> Dict[str, Any]:
    """加载票据规则，返回与ticket_rules.json匹配的规则字典"""
    with open(json_path, "r", encoding="utf-8") as f:
        raw_rules = json.load(f)

    ticket_rules = {}
    for ticket_type, raw_rule in raw_rules.items():
        # 构建格式检查函数
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

        # 构建逻辑检查函数
        logic_cond = raw_rule["logic_check_conditions"]

        def build_logic_check(min_count: int, field_list: List[str]):
            def logic_check(struct: Dict[str, Any]) -> bool:
                existing_fields = struct.get("key_fields_exist", [])
                matched_count = sum(1 for f in field_list if f in existing_fields)
                return matched_count >= min_count

            return logic_check

        # 组装规则（严格对应ticket_rules中的字段）
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
    """加载动态提示词，确保包含ticket_rules中的所有类型"""
    with open(json_path, "r", encoding="utf-8") as f:
        prompts = json.load(f)
    # 补充ticket_rules中存在但提示词中缺失的类型
    required_types = list(ticket_rules.keys()) + ["未知票据"]
    for t in required_types:
        if t not in prompts:
            prompts[t] = []
    return prompts


# 加载规则（核心：使用你的ticket_rules.json路径）
ticket_rules = load_ticket_rules_from_json(config.rule_json_path)
dynamic_prompts = load_dynamic_prompts(config.prompt_json_path)


# ------------------------------
# 3. 规则匹配函数（适配ticket_rules中的类型）
# ------------------------------
def rule_based_classify(ocr_text: str, structure: Dict[str, Any]) -> Tuple[str, float]:
    max_score = 0.0
    result_type = "未知票据"

    # 提取所有规则中定义的关键字段，检查OCR中存在哪些
    all_possible_fields = list(set(sum([r["key_fields"] for r in ticket_rules.values()], [])))
    existing_fields = [field for field in all_possible_fields if field in ocr_text]
    structure["key_fields_exist"] = existing_fields

    # 遍历ticket_rules中的所有票据类型进行匹配
    for ticket_type, rules in ticket_rules.items():
        score = 0.0
        type_prompts = dynamic_prompts.get(ticket_type, [])

        # 1. 核心名称匹配（ticket_rules中的core_name）
        core_match = any(name in ocr_text for name in rules["core_name"])
        if not core_match:
            continue
        score += 0.3

        # 2. 关键字段匹配（ticket_rules中的key_fields）
        base_field_matched = sum(1 for f in rules["key_fields"] if f in existing_fields)
        prompt_field_matched = sum(1 for f in existing_fields if f in type_prompts)
        total_field_score = (base_field_matched + min(prompt_field_matched, 2) * 0.5) / len(rules["key_fields"])
        field_contribution = min(total_field_score * 0.4, 0.4)
        score += field_contribution

        # 3. 格式特征匹配（ticket_rules中的format_check_conditions）
        base_format_ok = rules["format_check"](ocr_text, structure)
        if base_format_ok:
            base_format_score = 0.3
            struct_visual_feats = structure.get("image_features", []) + [
                structure.get("layout", ""), structure.get("background_color", ""), structure.get("shape", "")
            ]
            visual_prompt_matched = sum(1 for feat in struct_visual_feats if feat in type_prompts)
            visual_add = min(visual_prompt_matched * 0.05, 0.1)
            format_contribution = min(base_format_score + visual_add, 0.3)
            score += format_contribution

        # 4. 逻辑校验（ticket_rules中的logic_check_conditions）
        if not rules["logic_check"](structure):
            continue

        score = min(score, 1.0)
        if score > max_score and score >= config.confidence_threshold:
            max_score = score
            result_type = ticket_type

    return (result_type, round(max_score, 3))


# ------------------------------
# 4. 数据加载与预处理（确保标签与ticket_rules匹配）
# ------------------------------
def load_training_data(data_dir: str) -> Tuple[List[str], List[str], List[Dict], np.ndarray, LabelEncoder]:
    """加载训练数据，检查标签是否覆盖ticket_rules中的所有类型"""
    df = pd.read_csv(os.path.join(data_dir, "train.csv"))
    image_paths = [os.path.join(data_dir, p) for p in df["image_path"].values]
    ocr_texts = df["ocr_text"].values.tolist()
    structures = df["structure"].apply(eval).tolist()
    labels = df["label"].values.tolist()

    # 检查训练标签是否包含ticket_rules中的所有类型
    rule_types = set(ticket_rules.keys())
    train_types = set(labels)
    missing_types = rule_types - train_types
    if missing_types:
        raise ValueError(f"训练数据缺少ticket_rules中定义的类型：{missing_types}，请补充样本")

    # 标签编码（与ticket_rules类型对应）
    le = LabelEncoder()
    encoded_labels = le.fit_transform(labels)
    return image_paths, ocr_texts, structures, encoded_labels, le


def preprocess_image(img_path: str) -> tf.Tensor:
    """图像预处理：Resize+归一化"""
    img = Image.open(img_path).convert("RGB")
    img = img.resize(config.image_size)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    return img_array / 255.0  # 归一化到[0,1]


def create_text_tokenizer(texts: List[str]) -> Tokenizer:
    """创建文本Tokenizer（适配关键字段）"""

    def tokenize(text):
        return text.strip().split()

    tokenized_texts = [tokenize(t) for t in texts]
    tokenizer = Tokenizer(filters="", oov_token="<OOV>")
    tokenizer.fit_on_texts(tokenized_texts)
    return tokenizer


def create_meta_task(image_paths: List[str], ocr_texts: List[str],
                     encoded_labels: np.ndarray, tokenizer: Tokenizer) -> Tuple[Tuple, Tuple]:
    """生成少样本元学习任务"""
    unique_labels = np.unique(encoded_labels)
    sampled_labels = np.random.choice(
        unique_labels, size=config.num_ways, replace=len(unique_labels) < config.num_ways
    )

    support_imgs, support_texts, support_labels = [], [], []
    query_imgs, query_texts, query_labels = [], [], []

    for label in sampled_labels:
        label_idxs = np.where(encoded_labels == label)[0]
        required = config.num_support + config.num_query
        selected_idxs = np.random.choice(
            label_idxs, size=required, replace=len(label_idxs) < required
        )

        support_idxs = selected_idxs[:config.num_support]
        query_idxs = selected_idxs[config.num_support:]

        for idx in support_idxs:
            support_imgs.append(preprocess_image(image_paths[idx]))
            tokenized = ocr_texts[idx].strip().split()
            seq = tokenizer.texts_to_sequences([tokenized])[0]
            support_texts.append(pad_sequences([seq], maxlen=config.max_text_len)[0])
            support_labels.append(label)

        for idx in query_idxs:
            query_imgs.append(preprocess_image(image_paths[idx]))
            tokenized = ocr_texts[idx].strip().split()
            seq = tokenizer.texts_to_sequences([tokenized])[0]
            query_texts.append(pad_sequences([seq], maxlen=config.max_text_len)[0])
            query_labels.append(label)

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
    return [create_meta_task(image_paths, ocr_texts, encoded_labels, tokenizer)
            for _ in range(config.meta_batch_size)]


# 加载训练数据（含ticket_rules类型检查）
image_paths, ocr_texts, structures, encoded_labels, label_encoder = load_training_data(config.data_dir)
text_tokenizer = create_text_tokenizer(ocr_texts)
vocab_size = len(text_tokenizer.word_index) + 1


# ------------------------------
# 5. 多模态模型（CNN+LSTM）
# ------------------------------
def build_lstm_multimodal_model(vocab_size: int) -> models.Model:
    """构建适配Mac的轻量多模态模型"""
    # 图像分支（CNN）
    img_input = layers.Input(shape=(*config.image_size, 3), name="image_input")
    x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(img_input)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(128, (3, 3), activation="relu", padding="same")(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(256, (3, 3), activation="relu", padding="same")(x)
    img_feat = layers.GlobalAveragePooling2D()(x)
    img_feat = layers.BatchNormalization()(img_feat)

    # 文本分支（LSTM）
    text_input = layers.Input(shape=(config.max_text_len,), name="text_input")
    text_feat = layers.Embedding(
        input_dim=vocab_size,
        output_dim=config.embedding_dim,
        input_length=config.max_text_len,
        mask_zero=True
    )(text_input)
    text_feat = layers.LSTM(units=config.lstm_units, return_sequences=False)(text_feat)
    text_feat = layers.Dense(128, activation="relu")(text_feat)

    # 特征融合
    fused_feat = layers.Concatenate()([img_feat, text_feat])
    fused_feat = layers.Dense(256, activation="relu")(fused_feat)
    fused_feat = layers.Dropout(0.2)(fused_feat)

    return models.Model(inputs=[img_input, text_input], outputs=fused_feat)


# ------------------------------
# 6. 特征值字典（类别原型）生成与加载
# ------------------------------
def generate_class_prototypes(model: models.Model,
                              image_paths: List[str],
                              ocr_texts: List[str],
                              encoded_labels: np.ndarray,
                              tokenizer: Tokenizer,
                              label_encoder: LabelEncoder) -> Dict[int, np.ndarray]:
    """生成与ticket_rules匹配的特征值字典（每个类型的原型特征）"""
    class_prototypes = {}
    # 遍历ticket_rules中的所有类型
    for type_name in ticket_rules.keys():
        # 获取该类型的编码索引
        type_idx = label_encoder.transform([type_name])[0]
        # 找到该类型的所有样本索引
        sample_idxs = np.where(encoded_labels == type_idx)[0]
        if len(sample_idxs) == 0:
            raise ValueError(f"类型 {type_name} 无训练样本，无法生成原型")

        # 提取所有样本的特征并计算平均值（原型）
        feats = []
        for idx in sample_idxs:
            img_tensor = preprocess_image(image_paths[idx])
            img_tensor = tf.expand_dims(img_tensor, axis=0)
            tokenized = ocr_texts[idx].strip().split()
            text_seq = tokenizer.texts_to_sequences([tokenized])[0]
            text_seq = pad_sequences([text_seq], maxlen=config.max_text_len)
            feat = model([img_tensor, text_seq])  # 模型输出的融合特征
            feats.append(feat.numpy()[0])

        prototype = np.mean(feats, axis=0)  # 计算原型（平均特征）
        class_prototypes[type_idx] = prototype  # 用编码索引作为key
    return class_prototypes


def load_or_generate_prototypes(model: models.Model,
                                tokenizer: Tokenizer,
                                label_encoder: LabelEncoder) -> Tuple[tf.Tensor, tf.Tensor]:
    """加载已保存的特征值字典，若不存在则生成并保存"""
    if os.path.exists(config.prototype_save_path):
        # 加载已有的特征值字典
        class_prototypes = np.load(config.prototype_save_path, allow_pickle=True).item()
    else:
        # 生成新的特征值字典并保存
        class_prototypes = generate_class_prototypes(
            model=model,
            image_paths=image_paths,
            ocr_texts=ocr_texts,
            encoded_labels=encoded_labels,
            tokenizer=tokenizer,
            label_encoder=label_encoder
        )
        np.save(config.prototype_save_path, class_prototypes)

    # 转换为Tensor供模型使用
    prototypes = tf.convert_to_tensor(list(class_prototypes.values()), dtype=tf.float32)
    prototype_labels = tf.convert_to_tensor(list(class_prototypes.keys()), dtype=tf.int32)
    return prototypes, prototype_labels


# ------------------------------
# 7. 元学习训练与预测辅助函数
# ------------------------------
def compute_class_prototypes(features: tf.Tensor, labels: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    """计算类别原型（同类特征平均值）"""
    unique_labels = tf.unique(labels)[0]
    prototypes = []
    for label in unique_labels:
        class_feats = tf.boolean_mask(features, tf.equal(labels, label))
        prototypes.append(tf.reduce_mean(class_feats, axis=0))
    return tf.stack(prototypes, axis=0), unique_labels


def compute_euclidean_distance(features: tf.Tensor, prototypes: tf.Tensor) -> tf.Tensor:
    """计算特征与原型的欧氏距离"""
    feat_expand = tf.expand_dims(features, axis=1)
    proto_expand = tf.expand_dims(prototypes, axis=0)
    return tf.sqrt(tf.reduce_sum(tf.square(feat_expand - proto_expand), axis=-1) + 1e-6)


def meta_train_step(model: models.Model, meta_task: Tuple[Tuple, Tuple],
                    inner_lr: float, num_inner_steps: int) -> Tuple[tf.Tensor, List[tf.Tensor]]:
    """元学习单步训练"""
    (support_imgs, support_texts, support_labels), (query_imgs, query_texts, query_labels) = meta_task
    initial_weights = model.get_weights()
    inner_optimizer = optimizers.legacy.Adam(learning_rate=inner_lr)  # 适配Mac
    loss_fn = losses.SparseCategoricalCrossentropy(from_logits=True)

    # 内循环：适应支持集
    for _ in range(num_inner_steps):
        with tf.GradientTape() as tape:
            support_feats = model([support_imgs, support_texts], training=True)
            support_protos, unique_support_labels = compute_class_prototypes(support_feats, support_labels)
            support_distances = compute_euclidean_distance(support_feats, support_protos)
            support_logits = -support_distances

            label_mapping = tf.lookup.StaticHashTable(
                tf.lookup.KeyValueTensorInitializer(unique_support_labels,
                                                    tf.range(tf.shape(unique_support_labels)[0])),
                default_value=-1
            )
            mapped_support_labels = label_mapping.lookup(support_labels)
            support_loss = loss_fn(mapped_support_labels, support_logits + 1e-8)

        grads = tape.gradient(support_loss, model.trainable_variables)
        inner_optimizer.apply_gradients(zip(grads, model.trainable_variables))

    # 外循环：计算元损失
    with tf.GradientTape() as tape:
        query_feats = model([query_imgs, query_texts], training=True)
        query_distances = compute_euclidean_distance(query_feats, support_protos)
        query_logits = -query_distances
        mapped_query_labels = label_mapping.lookup(query_labels)
        query_loss = loss_fn(mapped_query_labels, query_logits + 1e-8)

    model.set_weights(initial_weights)  # 恢复初始权重
    meta_grads = tape.gradient(query_loss, model.trainable_variables)
    return query_loss, meta_grads


def run_meta_training(model: models.Model, image_paths: List[str], ocr_texts: List[str],
                      encoded_labels: np.ndarray, tokenizer: Tokenizer):
    """执行元学习训练"""
    outer_optimizer = optimizers.legacy.Adam(learning_rate=config.outer_lr)
    print("开始元学习训练...")
    for epoch in range(config.num_epochs):
        total_meta_loss = 0.0
        meta_batch = create_meta_batch(image_paths, ocr_texts, encoded_labels, tokenizer)
        for task in meta_batch:
            task_loss, task_grads = meta_train_step(
                model, task, config.inner_lr, config.num_inner_steps
            )
            outer_optimizer.apply_gradients(zip(task_grads, model.trainable_variables))
            total_meta_loss += task_loss.numpy()
        avg_meta_loss = total_meta_loss / len(meta_batch)
        print(f"Epoch {epoch + 1:2d}/{config.num_epochs} | 平均元损失：{avg_meta_loss:.4f}")
    print("元学习训练完成！")


# ------------------------------
# 8. 融合预测函数（使用特征值字典）
# ------------------------------
def predict_ticket_type(model: models.Model, img_path: str, ocr_text: str,
                        structure: Dict[str, Any], tokenizer: Tokenizer,
                        label_encoder: LabelEncoder, prototypes: tf.Tensor,
                        prototype_labels: tf.Tensor) -> Dict[str, Any]:
    """预测票据类型：模型（特征值字典匹配）+ 规则"""
    # 1. 输入预处理
    img_tensor = preprocess_image(img_path)
    img_tensor = tf.expand_dims(img_tensor, axis=0)
    tokenized_text = ocr_text.strip().split()
    text_seq = tokenizer.texts_to_sequences([tokenized_text])[0]
    text_seq = pad_sequences([text_seq], maxlen=config.max_text_len)

    # 2. 模型预测（基于特征值字典）
    model.trainable = False
    sample_feat = model([img_tensor, text_seq])  # 样本特征

    # 计算与所有原型的距离
    distances = compute_euclidean_distance(sample_feat, prototypes)
    min_dist_idx = tf.argmin(distances, axis=1).numpy()[0]  # 最近原型索引
    model_pred_label_idx = prototype_labels[min_dist_idx].numpy()  # 对应类别索引

    # 计算置信度
    min_distance = distances.numpy()[0][min_dist_idx]
    max_possible_distance = 10.0  # 可根据训练数据调整
    model_confidence = 1.0 - min(min_distance / max_possible_distance, 1.0)
    model_confidence = round(float(model_confidence), 3)

    # 转换为类型名称（与ticket_rules对应）
    model_pred_label = label_encoder.inverse_transform([model_pred_label_idx])[0]

    # 3. 规则匹配预测
    rule_pred_label, rule_confidence = rule_based_classify(ocr_text, structure)

    # 4. 融合结果
    if rule_confidence >= config.confidence_threshold:
        return {
            "票据类型": rule_pred_label,
            "置信度": rule_confidence,
            "分类来源": "规则匹配（优先）"
        }
    elif model_confidence >= config.confidence_threshold:
        return {
            "票据类型": model_pred_label,
            "置信度": model_confidence,
            "分类来源": "LSTM多模态模型"
        }
    else:
        return {
            "票据类型": "未知票据",
            "置信度": 0.0,
            "分类来源": "字段名/视觉特征不足，模型与规则均未匹配"
        }


# ------------------------------
# 9. 测试入口
# ------------------------------
def load_test_cases(json_path: str) -> List[Dict[str, Any]]:
    """加载测试用例"""
    with open(json_path, "r", encoding="utf-8") as f:
        test_data = json.load(f)
    return test_data.get("test_cases", [])


def run_ticket_test(model: models.Model, test_cases: List[Dict[str, Any]],
                    tokenizer: Tokenizer, label_encoder: LabelEncoder,
                    prototypes: tf.Tensor, prototype_labels: tf.Tensor):
    """执行测试"""
    print("\n" + "=" * 50)
    print("开始票据分类测试（基于ticket_rules）")
    print("=" * 50)
    for idx, case in enumerate(test_cases, 1):
        print(f"\n【测试用例 {idx}】{case.get('case_name', '未命名用例')}")
        img_relative_path = case.get("img_relative_path", "")
        ocr_text = case.get("ocr_text", "")
        structure = case.get("structure", {})
        img_abs_path = os.path.join(config.data_dir, img_relative_path)

        try:
            result = predict_ticket_type(
                model, img_abs_path, ocr_text, structure,
                tokenizer, label_encoder, prototypes, prototype_labels
            )
            for k, v in result.items():
                print(f"  {k}：{v}")
        except Exception as e:
            print(f"  测试失败：{str(e)}")
    print("\n" + "=" * 50)
    print("测试完成")
    print("=" * 50)


# ------------------------------
# 主程序（训练+测试）
# ------------------------------
if __name__ == "__main__":
    # 1. 构建并训练模型
    multimodal_model = build_lstm_multimodal_model(vocab_size)
    print("多模态模型结构：")
    multimodal_model.summary()
    run_meta_training(multimodal_model, image_paths, ocr_texts, encoded_labels, text_tokenizer)

    # 2. 生成/加载特征值字典（与ticket_rules匹配）
    prototypes, prototype_labels = load_or_generate_prototypes(
        model=multimodal_model,
        tokenizer=text_tokenizer,
        label_encoder=label_encoder
    )
    print(f"已加载特征值字典，包含 {len(prototypes)} 个票据类型的原型特征")

    # 3. 加载测试用例并执行测试
    test_cases = load_test_cases(config.test_case_json_path)
    if not test_cases:
        print("未加载到测试用例，请检查test_cases.json文件！")
    else:
        run_ticket_test(
            multimodal_model, test_cases, text_tokenizer,
            label_encoder, prototypes, prototype_labels
        )