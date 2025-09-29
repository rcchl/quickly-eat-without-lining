import os
import cv2
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import time


# 数据集类（修复路径处理和图像读取问题）
class SingleInvoiceDataset(Dataset):
    def __init__(self, single_image_path, transform=None):
        # 标准化路径
        self.single_image_path = os.path.normpath(single_image_path)
        self.image_paths = []

        print(f"🔍 检查图像路径: {self.single_image_path}")
        print(f"🔍 文件是否存在: {os.path.exists(self.single_image_path)}")

        # 验证图像文件
        if not os.path.exists(self.single_image_path):
            raise FileNotFoundError(f"❌ 图像文件不存在：{self.single_image_path}")
        elif not os.path.isfile(self.single_image_path):
            raise ValueError(f"❌ 不是有效文件：{self.single_image_path}")

        # 检查文件扩展名
        valid_extensions = ('.jpg', '.png', '.jpeg', '.bmp', '.tiff', '.webp')
        file_ext = os.path.splitext(self.single_image_path)[1].lower()
        if file_ext not in valid_extensions:
            raise ValueError(f"❌ 不支持的图像格式 {file_ext}，支持格式: {valid_extensions}")

        # 验证文件可读取
        try:
            # 尝试用OpenCV直接读取验证
            test_img = cv2.imread(self.single_image_path)
            if test_img is None:
                raise ValueError("OpenCV无法读取该文件")

            self.image_paths.append(self.single_image_path)
            print(f"✅ 图像验证成功：{os.path.basename(self.single_image_path)}")
            print(f"✅ 图像尺寸：{test_img.shape}")

        except Exception as e:
            raise ValueError(f"❌ 图像文件损坏或无法读取：{str(e)}")

        self.transform = transform
        # 简化标签逻辑：如果文件名包含"正常"则为正常发票，否则为异常发票
        filename = os.path.basename(self.single_image_path).lower()
        if "正常" in filename:
            self.labels = [0]  # 正常发票
            print(f"✅ 根据文件名标注为：正常发票")
        else:
            self.labels = [1]  # 异常发票
            print(f"✅ 根据文件名标注为：异常发票")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img_name = os.path.basename(img_path)

        try:
            # 使用OpenCV直接读取图像（更可靠）
            image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # 直接读取为灰度图
            if image is None:
                raise ValueError("cv2.imread返回None，文件可能损坏")

            # 图像预处理
            image = cv2.resize(image, (224, 224))  # 调整尺寸
            image = image.astype(np.float32) / 255.0  # 归一化到[0,1]
            image = (image - 0.5) / 0.5  # 标准化到[-1,1]
            image = np.expand_dims(image, axis=0)  # (1, 224, 224)

            label = self.labels[idx]

            # 转换为tensor
            image_tensor = torch.from_numpy(image).float()

            return image_tensor, torch.tensor(label, dtype=torch.long), img_path

        except Exception as e:
            print(f"❌ 图像处理失败 {img_name}: {str(e)}")
            # 返回一个空白图像作为fallback
            blank_image = np.random.rand(1, 224, 224).astype(np.float32)
            blank_image = (blank_image - 0.5) / 0.5
            return torch.from_numpy(blank_image).float(), torch.tensor(0, dtype=torch.long), img_path


# 模拟DeepSeek教师模型（修复逻辑）
class MockDeepSeekTeacherModel:
    def __init__(self):
        print("ℹ️ 使用【模拟DeepSeek教师模型】")
        self.noise = 0.05  # 减少噪声

    def get_teacher_logits(self, image_path):
        img_name = os.path.basename(image_path)
        print(f"ℹ️ 模拟DeepSeek推理：{img_name}")

        # 更智能的模拟逻辑
        filename_lower = img_name.lower()

        if "正常" in filename_lower or "proper" in filename_lower or "good" in filename_lower:
            base_logits = torch.tensor([3.0, 0.5], dtype=torch.float32)  # 强偏向正常
        elif "异常" in filename_lower or "abnormal" in filename_lower or "bad" in filename_lower:
            base_logits = torch.tensor([0.5, 3.0], dtype=torch.float32)  # 强偏向异常
        else:
            # 中性判断
            base_logits = torch.tensor([1.5, 1.5], dtype=torch.float32)

        # 添加轻微随机噪声
        random_noise = torch.randn_like(base_logits) * self.noise
        final_logits = base_logits + random_noise

        # 限制logits范围
        final_logits = torch.clamp(final_logits, min=0.1, max=4.0)

        pred_class = "正常发票" if final_logits[0] > final_logits[1] else "异常发票"
        print(f"ℹ️ 模拟DeepSeek预测：{pred_class}（logits: {final_logits.tolist()}）")

        return final_logits.unsqueeze(0)


# 简化学生模型（减少复杂度）
class LightweightStudentModel(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        # 更简单的网络结构
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.classifier(x)
        return x


# 知识蒸馏模块（修复设备兼容性）
class KnowledgeDistillationModule(nn.Module):
    def __init__(self, teacher_model, student_model, kd_loss_coef=1.0):
        super().__init__()
        self.teacher = teacher_model
        self.student = student_model
        self.kd_loss_coef = kd_loss_coef
        self.ce_loss = nn.CrossEntropyLoss()
        self.temperature = 3.0  # 降低温度参数

    def forward(self, x, labels, img_path):
        student_logits = self.student(x)

        # 获取教师logits（不传入设备，在外部处理）
        teacher_logits = self.teacher.get_teacher_logits(img_path[0] if isinstance(img_path, list) else img_path)

        # 确保teacher_logits在正确的设备上
        if hasattr(x, 'device'):
            teacher_logits = teacher_logits.to(x.device)

        # 硬标签损失
        hard_loss = self.ce_loss(student_logits, labels)

        # 知识蒸馏损失
        student_soft = F.log_softmax(student_logits / self.temperature, dim=1)
        teacher_soft = F.softmax(teacher_logits / self.temperature, dim=1)
        kd_loss = F.kl_div(student_soft, teacher_soft, reduction='batchmean') * (self.temperature ** 2)

        total_loss = hard_loss + self.kd_loss_coef * kd_loss
        return total_loss, student_logits, teacher_logits


# 简化指标计算（移除torchmetrics依赖）
def calculate_metrics(predictions, targets):
    """手动计算指标"""
    if len(predictions) == 0:
        return 0.0, 0.0, 0.0, 0.0

    pred_np = np.array(predictions)
    target_np = np.array(targets)

    accuracy = np.mean(pred_np == target_np)

    # 简化计算：对于二分类问题
    tp = np.sum((pred_np == 1) & (target_np == 1))
    fp = np.sum((pred_np == 1) & (target_np == 0))
    fn = np.sum((pred_np == 0) & (target_np == 1))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return accuracy, precision, recall, f1


# 主函数（全面修复）
def invoice_recognition_single_image(single_image_path=None, output_json=None):
    print("===== 开始单张发票识别流程（模拟DeepSeek） =====")

    # 设置默认路径
    if single_image_path is None:
        single_image_path = "C:/Users/zzh/Desktop/wechat_2025-09-29_212608_948.png"
    if output_json is None:
        output_json = "C:/Users/zzh/Desktop/单张发票识别结果.json"

    print(f"目标图像: {single_image_path}")
    print(f"输出文件: {output_json}")

    # 设备初始化
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 1. 加载单张图像数据集
    try:
        dataset = SingleInvoiceDataset(single_image_path)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
        print(f"✅ 成功加载图像，开始处理...")
    except Exception as e:
        print(f"❌ 数据集加载失败: {str(e)}")
        return None

    # 2. 初始化模型
    try:
        teacher_model = MockDeepSeekTeacherModel()
        student_model = LightweightStudentModel(num_classes=2).to(device)
        distillation_model = KnowledgeDistillationModule(teacher_model, student_model).to(device)
        print("✅ 模型初始化成功")
    except Exception as e:
        print(f"❌ 模型初始化失败: {str(e)}")
        return None

    # 3. 推理过程
    distillation_model.eval()
    recognition_results = []

    # 手动记录指标
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for batch_idx, (data, labels, img_paths) in enumerate(dataloader):
            img_path = img_paths[0] if isinstance(img_paths, (list, tuple)) else img_paths
            img_name = os.path.basename(img_path)
            print(f"\n----- 处理图像: {img_name} -----")

            try:
                data, labels = data.to(device), labels.to(device)
                print(f"✅ 数据加载到设备: data.shape={data.shape}, label={labels.item()}")

                start_time = time.time()

                # 核心推理
                total_loss, student_logits, teacher_logits = distillation_model(data, labels, img_path)
                infer_time = round(time.time() - start_time, 4)

                print(f"✅ 推理完成: loss={total_loss.item():.4f}, time={infer_time}s")

                # 解析输出
                student_probs = F.softmax(student_logits, dim=1).cpu().numpy()[0]
                pred_class_idx = torch.argmax(student_logits, dim=1).item()
                pred_class = "正常发票" if pred_class_idx == 0 else "异常发票"
                true_class = "正常发票" if labels.item() == 0 else "异常发票"

                # 记录预测结果
                all_predictions.append(pred_class_idx)
                all_targets.append(labels.item())

                # 保存结果
                recognition_results.append({
                    "图像文件名": img_name,
                    "图像路径": str(img_path),
                    "预测类别": pred_class,
                    "真实类别": true_class,
                    "预测概率": {
                        "正常发票": round(float(student_probs[0]), 4),
                        "异常发票": round(float(student_probs[1]), 4)
                    },
                    "推理时间(秒)": infer_time,
                    "学生模型logits": student_logits.cpu().numpy()[0].tolist(),
                    "教师模型logits": teacher_logits.cpu().numpy()[0].tolist()
                })

                print(f"✅ 识别成功: 预测={pred_class} | 真实={true_class}")
                print(f"   概率分布: 正常={student_probs[0]:.3f}, 异常={student_probs[1]:.3f}")

            except Exception as e:
                print(f"❌ 处理失败: {str(e)}")
                import traceback
                traceback.print_exc()
                continue

    # 4. 计算指标
    accuracy, precision, recall, f1 = calculate_metrics(all_predictions, all_targets)

    print(f"\n===== 处理完成 =====")
    print(f"✅ 成功处理 {len(recognition_results)}/1 张图像")
    print("📊 识别性能指标：")
    print(f"   准确率: {accuracy * 100:.2f}%")
    print(f"   精确率: {precision * 100:.2f}%")
    print(f"   召回率: {recall * 100:.2f}%")
    print(f"   F1分数: {f1 * 100:.2f}%")

    # 5. 生成并保存JSON结果
    final_result = {
        "模型信息": {
            "教师模型": "模拟DeepSeek视觉分类模型",
            "学生模型": "轻量级CNN",
            "任务类型": "发票二分类（正常/异常）",
            "使用设备": str(device)
        },
        "性能指标": {
            "总图像数": 1,
            "成功识别数": len(recognition_results),
            "准确率(%)": round(accuracy * 100, 2),
            "精确率(%)": round(precision * 100, 2),
            "召回率(%)": round(recall * 100, 2),
            "F1分数(%)": round(f1 * 100, 2)
        },
        "单图识别结果": recognition_results,
        "生成时间": time.strftime("%Y-%m-%d %H:%M:%S")
    }

    # 打印JSON摘要
    json_str = json.dumps(final_result, ensure_ascii=False, indent=2)
    print("\n===== 识别结果 =====")
    print(json_str)

    # 保存结果到文件
    try:
        output_dir = os.path.dirname(output_json)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            print(f"ℹ️ 创建输出目录: {output_dir}")

        with open(output_json, 'w', encoding='utf-8') as f:
            f.write(json_str)

        print(f"✅ 结果已保存至: {os.path.abspath(output_json)}")
    except Exception as e:
        print(f"❌ 保存结果失败: {str(e)}")
        print("ℹ️ 完整结果已打印在控制台")

    return final_result


if __name__ == "__main__":
    # 运行单张图像识别
    invoice_recognition_single_image(
        single_image_path="C:/Users/zzh/Desktop/wechat_2025-09-29_212608_948.png",
        output_json="C:/Users/zzh/Desktop/wechat_2025-09-29_212608_948_识别结果.json"
    )