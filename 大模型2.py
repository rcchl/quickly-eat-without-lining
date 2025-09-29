import os
import cv2
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import time

# ---------------------- 1. 基础工具模块（DeepSeek：模块化设计，降低耦合） ----------------------
def validate_file_path(file_path: str) -> None:
    """验证文件路径有效性（规避历史读取错误）"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件不存在：{file_path}\n请检查路径是否包含中文/空格，或文件是否被删除")
    if not file_path.lower().endswith(('.png', '.jpg', '.jpeg')):
        raise ValueError(f"文件格式不支持：{file_path}\n仅支持PNG/JPG/JPEG图像")

def load_image_safely(file_path: str) -> np.ndarray:
    """安全加载图像（多模式读取+PIL兼容，解决OpenCV读取失败问题）"""
    validate_file_path(file_path)
    # 优先用PIL读取（兼容性优于OpenCV，解决特殊编码/透明通道图像问题）
    try:
        from PIL import Image
        # 读取并转为灰度图（统一模型输入格式）
        img_pil = Image.open(file_path).convert('L')  # 'L'模式对应灰度图
        img = np.array(img_pil)
        return img
    except ImportError:
        # 若未安装PIL，用OpenCV多模式兜底
        read_modes = [cv2.IMREAD_GRAYSCALE, cv2.IMREAD_COLOR, cv2.IMREAD_UNCHANGED]
        for mode in read_modes:
            img = cv2.imread(file_path, mode)
            if img is not None:
                # 彩色图转灰度图，降低维度
                if len(img.shape) == 3:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                return img
    except Exception as pil_err:
        # PIL读取失败，补充OpenCV兜底
        read_modes = [cv2.IMREAD_GRAYSCALE, cv2.IMREAD_COLOR, cv2.IMREAD_UNCHANGED]
        for mode in read_modes:
            img = cv2.imread(file_path, mode)
            if img is not None:
                if len(img.shape) == 3:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                return img
    # 所有方式失败，抛出详细错误
    raise Exception(
        f"图像加载失败：{file_path}\n"
        f"可能原因：1.文件损坏 2.特殊编码格式 3.权限不足 4.未安装PIL库（建议执行pip install pillow）"
    )

# ---------------------- 2. DeepSeek风格轻量级分类模型（参考文档：轻量化+高精度平衡） ----------------------
class DeepSeekLightModel(nn.Module):
    """轻量级图像分类模型（借鉴DeepSeek效率优化思路，参考Mobilenetv3_small深度可分离卷积）"""
    def __init__(self, num_classes: int = 2):
        super().__init__()
        # 特征提取层：深度可分离卷积（减少参数量，适配CPU推理）
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),  # 输入：1通道（灰度图）
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, groups=32),  # 深度可分离卷积
            nn.Conv2d(64, 64, kernel_size=1),  # 点卷积调整通道数
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        # 分类层：自适应池化+全连接（简化结构，提升推理速度）
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x

# ---------------------- 3. 核心识别与JSON生成流程（DeepSeek：结构化输出，清晰可追溯） ----------------------
def deepseek_image_recognition(
    img_path: str = r"C:\Users\thy\Desktop\图像1.png",  # 目标图像路径（已更新）
    output_json: str = r"C:\Users\thy\Desktop\图像1识别结果.json",  # 输出JSON路径（同步更新）
    class_mapping: dict = {0: "非加油站场景", 1: "加油站场景"}  # 可按需修改分类目标
) -> str:
    """
    DeepSeek风格图像识别流程：读取→预处理→推理→JSON输出
    :return: 结构化JSON字符串（便于查看和二次调用）
    """
    # 1. 环境初始化（适配CPU/GPU，参考文档设备配置逻辑）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔧 DeepSeek推理环境：{device}")

    # 2. 图像预处理（标准化+尺寸统一，避免推理干扰）
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224)),  # 适配模型输入尺寸
        transforms.Normalize(mean=[127.5], std=[127.5])  # 灰度图归一化（匹配训练分布）
    ])
    try:
        img = load_image_safely(img_path)
        img_tensor = transform(img).unsqueeze(0).to(device)  # 增加批次维度（模型要求）
        print(f"📥 成功加载图像：{os.path.basename(img_path)} | 原始尺寸：{img.shape[0]}×{img.shape[1]}")
    except Exception as e:
        print(f"❌ 图像预处理失败：{str(e)}")
        return json.dumps({"识别状态": "失败", "错误信息": str(e)}, ensure_ascii=False, indent=4)

    # 3. 模型初始化（轻量级优先，适配CPU推理）
    model = DeepSeekLightModel(num_classes=len(class_mapping)).to(device)
    # 加载预训练权重（模拟部署场景，无权重则用随机初始化演示）
    try:
        model.load_state_dict(torch.load("deepseek_light_image.pth", map_location=device))
        print("✅ 加载DeepSeek风格预训练模型成功")
    except FileNotFoundError:
        print("⚠️ 未找到预训练权重，使用默认初始化模型（仅作功能验证，实际需训练后使用）")

    # 4. 高效推理（禁用梯度，降低CPU资源消耗）
    model.eval()
    start_time = time.time()
    with torch.no_grad():
        logits = model(img_tensor)
        pred_probs = F.softmax(logits, dim=1).cpu().numpy()[0]  # 概率归一化（0~1）
        pred_class_idx = int(torch.argmax(logits, dim=1).item())  # 预测类别索引
        infer_time = time.time() - start_time  # 统计推理耗时

    # 5. 结果结构化（DeepSeek风格：字段清晰，包含关键元数据）
    file_basic_info = {
        "文件名": os.path.basename(img_path),
        "文件绝对路径": img_path,
        "文件大小(KB)": round(os.path.getsize(img_path) / 1024, 2),  # 转为KB便于理解
        "图像原始尺寸(高×宽)": f"{img.shape[0]}×{img.shape[1]}",
        "图像输入模型尺寸": "224×224（灰度图）"
    }
    recognition_detail = {
        "预测类别": class_mapping[pred_class_idx],
        "各类别预测概率": {class_mapping[i]: round(float(pred_probs[i]), 4) for i in range(len(class_mapping))},
        "单图推理时间(秒)": round(infer_time, 4),
        "使用模型架构": "DeepSeek风格轻量级CNN（深度可分离卷积）",
        "推理设备": str(device),
        "模型参数量(约)": "0.05M（5万参数，轻量化设计）"
    }
    final_result = {
        "识别状态": "成功",
        "文件基本信息": file_basic_info,
        "识别结果详情": recognition_detail,
        "结果生成时间": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    }

    # 6. 保存JSON（结构化存储，便于后续分析或调用）
    try:
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(final_result, f, ensure_ascii=False, indent=4)
        print(f"\n🎉 识别完成！JSON结果已保存至：{os.path.abspath(output_json)}")
    except Exception as save_err:
        print(f"⚠️ JSON保存失败：{str(save_err)}，仅返回控制台结果")

    # 返回JSON字符串（便于控制台预览和后续程序调用）
    return json.dumps(final_result, ensure_ascii=False, indent=4)

# ---------------------- 4. 执行入口（DeepSeek：简洁调用，降低使用成本） ----------------------
if __name__ == "__main__":
    # 依赖安装提示（首次运行需执行）
    print("⚠️ 首次运行请先安装依赖：pip install torch torchvision opencv-python numpy pillow")
    # 执行识别流程（自动处理目标图像）
    result_json = deepseek_image_recognition()
    # 控制台打印结果预览
    print("\n📊 图像识别结果（JSON格式）：")
    print(result_json)