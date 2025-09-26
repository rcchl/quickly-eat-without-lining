import os
import json
import time
import win32com.client  # 需安装pywin32：pip install pywin32


class WpsOcrManager:
    def __init__(self, wps_dir):
        self.wps_dir = wps_dir  # WPS安装目录（含office6的路径）
        self.wps_app = None  # WPS应用实例
        self.callback = None  # 结果回调函数
        self.task_queue = []  # 任务队列

    def SetExePath(self, wps_exe_path):
        """设置WPS主程序路径（wps.exe所在路径）"""
        self.wps_exe_path = wps_exe_path

    def SetUsrLibDir(self, wps_dir):
        """设置WPS库目录（同安装目录）"""
        self.wps_dir = wps_dir

    def SetOcrResultCallback(self, callback):
        """设置识别结果回调函数"""
        self.callback = callback

    def StartWeChatOCR(self):  # 保持方法名与微信代码一致（兼容结构）
        """启动WPS应用（后台运行）"""
        try:
            # 启动WPS并隐藏窗口
            self.wps_app = win32com.client.Dispatch("KWPS.Application")
            self.wps_app.Visible = False  # 不显示WPS窗口
            print("WPS OCR服务启动成功")
        except Exception as e:
            print(f"WPS OCR服务启动失败：{e}")
            raise

    def DoOCRTask(self, img_path):
        """执行图片文字提取任务"""
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"图片不存在：{img_path}")

        try:
            # 新建空白文档
            doc = self.wps_app.Documents.Add()
            # 插入图片到文档
            shape = doc.Shapes.AddPicture(
                FileName=img_path,
                LinkToFile=False,
                SaveWithDocument=True
            )
            # 等待图片加载（根据图片大小调整延迟）
            time.sleep(1)

            # 执行「提取文字」（WPS的OCR功能，不同版本方法名可能不同）
            # 方法1：尝试直接调用OCR接口（适合部分版本）
            try:
                shape.OCR()  # 执行OCR识别
            except:
                # 方法2：若OCR()失败，尝试通过菜单命令触发「提取文字」
                self.wps_app.CommandBars.ExecuteMso("PictureExtractText")
                time.sleep(1)  # 等待提取完成

            # 提取识别结果（从文档中获取OCR后的文字）
            ocr_text = doc.Content.Text.strip()  # 获取文档所有内容（即OCR结果）

            # 构造与微信OCR格式一致的结果字典
            results = {
                "texts": [{"content": ocr_text, "position": [0, 0, 0, 0]}],  # 简化位置信息
                "img_path": img_path,
                "status": "success"
            }

            # 调用回调函数处理结果
            if self.callback:
                self.callback(img_path, results)
            self.task_queue.append(img_path)

            # 关闭文档（不保存）
            doc.Close(SaveChanges=0)
        except Exception as e:
            print(f"OCR识别失败：{e}")
            doc.Close(SaveChanges=0)

    def KillWeChatOCR(self):  # 保持方法名与微信代码一致（兼容结构）
        """关闭WPS应用"""
        if self.wps_app:
            self.wps_app.Quit()
            self.wps_app = None
            print("WPS OCR服务已关闭")


# 结果回调函数（与微信代码完全一致）
def ocr_result_callback(img_path: str, results: dict):
    result_file = os.path.basename(img_path) + ".json"
    print(f"识别成功，img_path: {img_path}, result_file: {result_file}")
    with open(result_file, 'w', encoding='utf-8') as f:
        f.write(json.dumps(results, ensure_ascii=False, indent=2))


def main():
    # WPS安装目录（需替换为实际路径，通常含office6）
    wps_dir = r"C:\Users\颖\AppData\Local\Kingsoft\WPS Office\12.1.0.22529\office6"
    # WPS主程序路径（wps.exe所在位置）
    wps_exe_path = os.path.join(wps_dir, "wps.exe")

    # 初始化WPS OCR管理器（结构与微信代码一致）
    ocr_manager = WpsOcrManager(wps_dir)
    ocr_manager.SetExePath(wps_exe_path)
    ocr_manager.SetUsrLibDir(wps_dir)
    ocr_manager.SetOcrResultCallback(ocr_result_callback)

    # 启动WPS OCR服务
    ocr_manager.StartWeChatOCR()

    # 执行图片文字提取任务（替换为你的图片路径）
    img_path = r"C:\Users\颖\Desktop\0BF9F651-FF22-C4F4-6503-8B1248C22EC4.jpg"
    ocr_manager.DoOCRTask(img_path)

    # 等待任务完成（根据实际情况调整延迟）
    time.sleep(2)

    # 关闭服务
    ocr_manager.KillWeChatOCR()


if __name__ == "__main__":
    main()