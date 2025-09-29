import os
import cv2
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import time
from sklearn.model_selection import train_test_split


# ---------------------- 1. 数据准备模块（新增：处理训练数据） ----------------------
class DocDataset(Dataset):
    """文档数据集类，用于加载和预处理训练图片"""

    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths  # 图片路径列表
        self.labels = labels  # 标签列表（0:发票, 1:资产负债表, 2:利润表, 3:营业执照）
        self.transform = transform  # 预处理方法

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # 读取图片并转为灰度图
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert('L')  # 转为灰度图
        label = self.labels[idx]

        # 应用预处理
        if self.transform:
            img = self.transform(img)
        return img, label


def prepare_train_data(data_dir):
    """
    准备训练数据
    :param data_dir: 数据文件夹，结构应为：
                     data_dir/
                        0_发票/
                        1_资产负债表/
                        2_利润表/
                        3_营业执照/
    """
    image_paths = []
    labels = []

    # 遍历每个类别文件夹
    for label in [0, 1, 2, 3]:
        class_dir = os.path.join(data_dir, f"{label}_{['发票', '资产负债表', '利润表', '营业执照'][label]}")
        if not os.path.exists(class_dir):
            raise FileNotFoundError(f"类别文件夹不存在：{class_dir}\n请按要求创建数据文件夹结构")

        # 收集该类别下的所有图片
        for img_name in os.listdir(class_dir):
            if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_paths.append(os.path.join(class_dir, img_name))
                labels.append(label)

    # 分割训练集和验证集（8:2）
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        image_paths, labels, test_size=0.2, random_state=42
    )

    # 定义训练集预处理（增加数据增强）
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomRotation(10),  # 随机旋转，增强泛化能力
        transforms.ColorJitter(contrast=2.0),  # 增强对比度，突出文字
        transforms.ToTensor(),
        transforms.Normalize(mean=[127.5], std=[127.5])
    ])

    # 验证集和推理用相同预处理
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[127.5], std=[127.5])
    ])

    # 创建数据集和数据加载器
    train_dataset = DocDataset(train_paths, train_labels, train_transform)
    val_dataset = DocDataset(val_paths, val_labels, val_transform)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=0)

    print(f"📊 数据准备完成 | 训练样本：{len(train_dataset)} | 验证样本：{len(val_dataset)}")
    return train_loader, val_loader


# ---------------------- 2. 模型定义（优化版：增强特征提取） ----------------------
class DeepSeekDocTypeModel(nn.Module):
    """优化版文档分类模型，增强对发票和利润表的区分能力"""

    def __init__(self, num_classes: int = 4):
        super().__init__()
        # 增强特征提取层：更关注文字和表格特征
        self.features = nn.Sequential(
            # 第一层：捕捉基础边缘特征
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # 第二层：深度可分离卷积，聚焦局部特征（如发票章、表格线）
            nn.Conv2d(32, 64, kernel_size=3, padding=1, groups=32),
            nn.Conv2d(64, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # 第三层：增强对细节特征的捕捉（如"发票"vs"利润表"文字）
            nn.Conv2d(64, 128, kernel_size=3, padding=1, groups=64),
            nn.Conv2d(128, 128, kernel_size=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # 新增第四层：强化对复杂特征的学习
            nn.Conv2d(128, 256, kernel_size=3, padding=1, groups=128),
            nn.Conv2d(256, 256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        # 分类层：更复杂的全连接网络
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),  # 防止过拟合
            nn.Linear(128, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x


# ---------------------- 3. 训练模块（新增：模型训练功能） ----------------------
def train_model(data_dir, epochs=20):
    """训练模型并保存权重"""
    # 准备数据
    train_loader, val_loader = prepare_train_data(data_dir)

    # 初始化模型、优化器和损失函数
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DeepSeekDocTypeModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()  # 分类问题常用损失函数

    best_val_acc = 0.0  # 记录最佳验证准确率

    for epoch in range(epochs):
        # 训练阶段
        model.train()  # 切换到训练模式
        train_loss = 0.0
        train_correct = 0
        total_train = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            # 前向传播
            outputs = model(images)
            loss = criterion(outputs, labels)

            # 反向传播和优化
            optimizer.zero_grad()  # 清空梯度
            loss.backward()  # 计算梯度
            optimizer.step()  # 更新参数

            # 统计训练指标
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            train_correct += (predicted == labels).sum().item()

        # 验证阶段
        model.eval()  # 切换到评估模式
        val_loss = 0.0
        val_correct = 0
        total_val = 0

        with torch.no_grad():  # 禁用梯度计算
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        # 计算准确率
        train_acc = 100 * train_correct / total_train
        val_acc = 100 * val_correct / total_val

        # 打印本轮结果
        print(f"\nEpoch {epoch + 1}/{epochs}")
        print(f"训练损失: {train_loss / len(train_loader):.4f} | 训练准确率: {train_acc:.2f}%")
        print(f"验证损失: {val_loss / len(val_loader):.4f} | 验证准确率: {val_acc:.2f}%")

        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "deepseek_doc_type_model.pth")
            print(f"✅ 保存最佳模型（验证准确率：{best_val_acc:.2f}%）")

    print(f"\n🏁 训练完成！最佳验证准确率：{best_val_acc:.2f}% | 模型已保存为 deepseek_doc_type_model.pth")
    return model


# ---------------------- 4. 推理模块（优化版：增加概率过滤） ----------------------
def deepseek_doc_type_recognition(
        img_path: str = r"C:\Users\thy\Desktop\图像1.png",
        output_json: str = r"C:\Users\thy\Desktop\贷款客户文档识别结果.json",
        doc_type_mapping: dict = {
            0: "发票",
            1: "资产负债表",
            2: "利润表",
            3: "营业执照"
        }
) -> str:
    """文档类型识别主函数（优化版：增加低概率过滤）"""
    # 1. 环境初始化
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔧 推理环境：{device} | 支持类型：{list(doc_type_mapping.values())}")

    # 2. 图像预处理（与验证集保持一致）
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[127.5], std=[127.5])
    ])

    try:
        # 读取图像
        img = Image.open(img_path).convert('L')  # 用PIL读取灰度图
        img_np = np.array(img)  # 用于记录尺寸信息
        img_tensor = transform(img).unsqueeze(0).to(device)
        print(f"📥 加载成功：{os.path.basename(img_path)} | 尺寸：{img_np.shape[0]}×{img_np.shape[1]}")
    except Exception as e:
        print(f"❌ 加载失败：{str(e)}")
        return json.dumps({"识别状态": "失败", "错误信息": str(e)}, ensure_ascii=False, indent=4)

    # 3. 模型加载
    model = DeepSeekDocTypeModel().to(device)
    try:
        model.load_state_dict(torch.load("deepseek_doc_type_model.pth", map_location=device))
        print("✅ 加载训练好的模型成功")
    except FileNotFoundError:
        print("⚠️ 未找到训练好的模型，使用随机模型（效果差！请先训练）")

    # 4. 推理计算（增加概率过滤）
    model.eval()
    start_time = time.time()
    with torch.no_grad():
        logits = model(img_tensor)
        pred_probs = F.softmax(logits, dim=1).cpu().numpy()[0]  # 概率归一化
        pred_type_idx = int(torch.argmax(logits, dim=1).item())
        pred_doc_type = doc_type_mapping[pred_type_idx]

        # 低概率过滤（低于60%标记为不确定）
        if max(pred_probs) < 0.6:
            pred_doc_type = f"无法确定（疑似{pred_doc_type}，建议人工复核）"

        infer_time = time.time() - start_time

    # 5. 结果整理
    final_result = {
        "识别状态": "成功",
        "文件信息": {
            "文件名": os.path.basename(img_path),
            "路径": img_path,
            "大小(KB)": round(os.path.getsize(img_path) / 1024, 2),
            "尺寸": f"{img_np.shape[0]}×{img_np.shape[1]}"
        },
        "识别结果": {
            "文档类型": pred_doc_type,
            "各类别概率": {doc_type_mapping[i]: round(float(pred_probs[i]), 4) for i in range(4)},
            "推理时间(秒)": round(infer_time, 4),
            "设备": str(device)
        },
        "生成时间": time.strftime("%Y-%m-%d %H:%M:%S")
    }

    # 6. 保存结果
    try:
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(final_result, f, ensure_ascii=False, indent=4)
        print(f"\n🎉 结果已保存至：{os.path.abspath(output_json)}")
    except Exception as e:
        print(f"⚠️ 保存失败：{str(e)}")

    return json.dumps(final_result, ensure_ascii=False, indent=4)


# ---------------------- 5. 执行入口 ----------------------
if __name__ == "__main__":
    # 首次运行请安装依赖
    print("⚠️ 首次运行请安装依赖：pip install torch torchvision opencv-python numpy pillow scikit-learn")

    # 步骤1：训练模型（请先按要求准备数据）
    # 数据文件夹路径（请修改为你的实际路径）
    data_directory = r"C:\Users\thy\Desktop\文档训练数据"  # 里面应有4个子文件夹（0_发票到3_营业执照）
    if not os.path.exists(data_directory):
        print("\n⚠️ 未找到训练数据文件夹，请按以下结构创建：")
        print(f"{data_directory}/")
        print("  ├─0_发票/       （存放发票图片）")
        print("  ├─1_资产负债表/ （存放资产负债表图片）")
        print("  ├─2_利润表/     （存放利润表图片）")
        print("  └─3_营业执照/   （存放营业执照图片）")
    else:
        print("\n📌 开始训练模型...")
        train_model(data_directory, epochs=20)  # 训练20轮（可调整）

    # 步骤2：识别文档（训练完成后执行）
    print("\n📌 开始识别文档...")
    result = deepseek_doc_type_recognition(
        img_path=r"C:\Users\thy\Desktop\图像1.png",  # 待识别的发票图片
        output_json=r"C:\Users\thy\Desktop\识别结果.json"
    )
    print("\n📊 识别结果：")
    print(result)
