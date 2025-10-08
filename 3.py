import os
import cv2
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models  # 引入torchvision预训练模型
import time


# 1. 数据集类（修改预处理以适配真实模型）
class SingleInvoiceDataset(Dataset):
    def __init__(self, single_image_path):
        self.single_image_path = os.path.normpath(single_image_path)
        self.image_paths = []
        self.labels = []

        # 验证图像路径（保留原有逻辑）
        if not os.path.exists(self.single_image_path):
            raise FileNotFoundError(f"❌ 图像文件不存在：{self.single_image_path}")
        if not os.path.isfile(self.single_image_path):
            raise ValueError(f"❌ 不是有效文件：{self.single_image_path}")

        # 验证图像格式
        valid_extensions = ('.jpg', '.png', '.jpeg')
        file_ext = os.path.splitext(self.single_image_path)[1].lower()
        if file_ext not in valid_extensions:
            raise ValueError(f"❌ 不支持的图像格式 {file_ext}")

        # 验证图像可读取
        test_img = cv2.imread(self.single_image_path)
        if test_img is None:
            raise ValueError("❌ OpenCV无法读取该文件")
        self.image_paths.append(self.single_image_path)
        print(f"✅ 图像验证成功：{os.path.basename(self.single_image_path)}，尺寸：{test_img.shape}")

        # 标签逻辑（保留）
        filename = os.path.basename(self.single_image_path).lower()
        self.labels = [0] if "正常" in filename else [1]
        print(f"✅ 标签：{'正常发票' if self.labels[0] == 0 else '异常发票'}")

        # ！！关键修改：真实模型的预处理（与模型训练时一致）
        self.transform = transforms.Compose([
            transforms.ToPILImage(),  # 转换为PIL图像（适配torchvision）
            transforms.Resize((224, 224)),  # 真实模型输入尺寸（需与模型一致）
            transforms.ToTensor(),  # 转换为Tensor（0-1）
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],  # ImageNet均值（预训练模型通用）
                std=[0.229, 0.224, 0.225]   # ImageNet标准差
            )
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # ！！关键：OpenCV默认BGR，转换为RGB（与模型一致）
        img_tensor = self.transform(img)  # 应用预处理
        return img_tensor, torch.tensor(self.labels[idx], dtype=torch.long), img_path


# 2. 真实教师模型（替换伪模型）
class RealTeacherModel:
    def __init__(self, model_path=None, device='cpu'):
        self.device = device
        # ！！关键：加载真实预训练模型
        self.model = self._load_model(model_path)
        self.model.to(self.device)
        self.model.eval()  # 设置为推理模式
        print(f"✅ 真实教师模型加载完成（设备：{self.device}）")

    def _load_model(self, model_path):
        # 示例：加载预训练ResNet-152（可替换为自定义微调模型）
        if model_path is None:
            # 若没有本地微调模型，使用torchvision预训练模型（需后续适配任务）
            model = models.resnet152(pretrained=True)
            # 替换最后一层以适配发票二分类任务
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, 2)  # 输出2类（正常/异常）
        else:
            # 加载本地微调过的模型（需确保模型结构匹配）
            model = models.resnet152(pretrained=False)
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, 2)
            model.load_state_dict(torch.load(model_path, map_location=self.device))
        return model

    def get_teacher_logits(self, img_tensor):
        # ！！关键：真实模型推理（输入为预处理后的图像张量）
        with torch.no_grad():  # 关闭梯度计算
            img_tensor = img_tensor.unsqueeze(0)  # 增加batch维度
            logits = self.model(img_tensor.to(self.device))  # 模型输出logits
        return logits  # 输出形状：(1, 2)


# 3. 学生模型和蒸馏模块（基本不变，适配输入）
class LightweightStudentModel(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),  # ！！输入改为3通道（RGB）
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


class KnowledgeDistillationModule(nn.Module):
    def __init__(self, teacher_model, student_model, kd_loss_coef=1.0):
        super().__init__()
        self.teacher = teacher_model
        self.student = student_model
        self.kd_loss_coef = kd_loss_coef
        self.ce_loss = nn.CrossEntropyLoss()
        self.temperature = 3.0

    def forward(self, x, labels):
        # ！！修改：教师模型直接接收图像张量（而非路径）
        student_logits = self.student(x)
        teacher_logits = self.teacher.get_teacher_logits(x)  # 真实模型推理

        # 损失计算（不变）
        hard_loss = self.ce_loss(student_logits, labels)
        student_soft = F.log_softmax(student_logits / self.temperature, dim=1)
        teacher_soft = F.softmax(teacher_logits / self.temperature, dim=1)
        kd_loss = F.kl_div(student_soft, teacher_soft, reduction='batchmean') * (self.temperature **2)

        total_loss = hard_loss + self.kd_loss_coef * kd_loss
        return total_loss, student_logits, teacher_logits


# 4. 主函数（适配真实模型的输入输出）
def invoice_recognition_single_image(single_image_path=None, model_path=None, output_json=None):
    print("===== 真实大模型发票识别流程 =====")
    # 路径配置
    if single_image_path is None:
        single_image_path = "C:/Users/zzh/Desktop/wechat_2025-09-29_212608_948.png"
    if output_json is None:
        output_json = "C:/Users/zzh/Desktop/真实模型识别结果.json"

    # 设备配置（优先GPU）
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 加载数据集（预处理已适配真实模型）
    try:
        dataset = SingleInvoiceDataset(single_image_path)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    except Exception as e:
        print(f"❌ 数据集加载失败: {str(e)}")
        return None

    # 初始化模型（真实教师模型+学生模型）
    try:
        teacher_model = RealTeacherModel(model_path=model_path, device=device)  # 加载真实模型
        student_model = LightweightStudentModel(num_classes=2).to(device)
        distillation_model = KnowledgeDistillationModule(teacher_model, student_model).to(device)
    except Exception as e:
        print(f"❌ 模型初始化失败: {str(e)}")
        return None

    # 推理
    distillation_model.eval()
    recognition_results = []
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for data, labels, img_paths in dataloader:
            img_path = img_paths[0]
            img_name = os.path.basename(img_path)
            print(f"\n----- 处理图像: {img_name} -----")

            data, labels = data.to(device), labels.to(device)
            start_time = time.time()

            # ！！推理时直接传入图像张量（真实模型不需要路径）
            total_loss, student_logits, teacher_logits = distillation_model(data, labels)
            infer_time = round(time.time() - start_time, 4)

            # 解析结果
            student_probs = F.softmax(student_logits, dim=1).cpu().numpy()[0]
            pred_class_idx = torch.argmax(student_logits, dim=1).item()
            pred_class = "正常发票" if pred_class_idx == 0 else "异常发票"
            true_class = "正常发票" if labels.item() == 0 else "异常发票"

            all_predictions.append(pred_class_idx)
            all_targets.append(labels.item())

            recognition_results.append({
                "图像文件名": img_name,
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

            print(f"✅ 识别成功: 预测={pred_class}，概率={student_probs[pred_class_idx]:.4f}")

    # 保存结果（逻辑不变）
    final_result = {
        "模型信息": {
            "教师模型": "真实ResNet-152（预训练/微调）",
            "学生模型": "轻量级CNN",
            "使用设备": str(device)
        },
        "性能指标": {
            "准确率(%)": round(np.mean(np.array(all_predictions) == np.array(all_targets)) * 100, 2)
        },
        "单图识别结果": recognition_results,
        "生成时间": time.strftime("%Y-%m-%d %H:%M:%S")
    }

    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(final_result, f, ensure_ascii=False, indent=2)
    print(f"\n✅ 结果保存至: {output_json}")
    return final_result


if __name__ == "__main__":
    # 若有本地微调模型，传入model_path；否则传None使用预训练模型
    invoice_recognition_single_image(
        single_image_path="C:/Users/zzh/Desktop/wechat_2025-09-29_212608_948.png",
        model_path=None,  # 替换为你的模型权重路径（如"./invoice_resnet152.pth"）
        output_json="C:/Users/zzh/Desktop/真实模型识别结果.json"
    )
