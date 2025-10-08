import time
import json
import base64
import requests
import io
from typing import Dict, List, Union
from PIL import Image  # 需安装：pip install pillow


class LoanMaterialOCR:
    """贷款材料通用OCR识别工具（支持进账单固定区域分区识别）"""

    def __init__(self, app_id: str, api_key: str, secret_key: str):
        # 手动填入的图片像素尺寸（核心：所有坐标基于此尺寸）
        self.image_width = 1379  # 需用户手动填入
        self.image_height = 668  # 需用户手动填入
        self.app_id = app_id
        self.api_key = api_key
        self.secret_key = secret_key
        self.access_token = ""  # 接口访问令牌
        self.material_type_map = {
            "票据类": ["支票", "进账单", "汇款单", "转账支票"],
            "报表类": ["利润表", "资产负债表", "现金流量表"]
        }
        # 进账单固定区域坐标（需根据实际图片红色框选位置修改！）
        # 格式：区域名称 -> (左上角x1, 左上角y1, 右下角x2, 右下角y2)
        self.BILL_TEMPLATE_AREAS = {
            "出票人户名": (295, 166, 686, 204),  # 出票人“全称”区域
            "出票人账号": (295, 207, 686, 244),  # 出票人“账号”区域
            "出票人开户行": (295, 253, 686, 305),  # 出票人“开户银行”区域
            "收款人户名": (890, 167, 1274, 209),  # 收款人“全称”区域
            "收款人账号": (890, 213, 1274, 260),  # 收款人“账号”区域
            "收款人开户行": (890, 259, 1274, 304),  # 收款人“开户银行”区域
            "金额大写": (148, 304,684, 380 ),  # “金额（大写）”区域
            "金额小写": (927, 338, 1271, 383 ),  # “金额（小写）”区域
            "票据日期": (457, 91, 913, 160)  # 日期“2021年1月25日”区域
        }
        # 初始化时获取令牌并验证凭证
        self._get_access_token()
        self._verify_credentials()

    def _get_access_token(self) -> bool:
        """获取百度OCR接口访问令牌（有效期30天）"""
        print("=== 正在获取接口访问令牌... ===")
        token_url = "https://aip.baidubce.com/oauth/2.0/token"
        params = {
            "grant_type": "client_credentials",
            "client_id": self.api_key,
            "client_secret": self.secret_key
        }
        try:
            response = requests.get(
                token_url,
                params=params,
                headers={"Content-Type": "application/json; charset=utf-8"}
            )
            response.encoding = "utf-8"
            result = response.json()

            if "access_token" in result:
                self.access_token = result["access_token"]
                print(f"✅ 令牌获取成功（有效期：{result.get('expires_in', 0) // 3600}小时）")
                return True
            else:
                error_msg = result.get("error_description", "未知错误")
                print(f"❌ 令牌获取失败：{error_msg}")
                self.access_token = ""
                return False
        except Exception as e:
            print(f"❌ 令牌获取异常：{str(e)}（检查网络连接或API_KEY/SECRET_KEY）")
            self.access_token = ""
            return False

    def _verify_credentials(self) -> bool:
        """验证凭证有效性（调用通用文字识别接口测试）"""
        if not self.access_token:
            print("❌ 凭证验证跳过：未获取到访问令牌")
            return False

        print("\n=== 正在验证百度OCR凭证有效性... ===")
        ocr_url = f"https://aip.baidubce.com/rest/2.0/ocr/v1/general_basic?access_token={self.access_token}"
        empty_image_base64 = base64.b64encode(b"").decode("utf-8")
        data = {"image": empty_image_base64}

        try:
            response = requests.post(
                ocr_url,
                data=data,
                headers={"Content-Type": "application/x-www-form-urlencoded; charset=utf-8"}
            )
            response.encoding = "utf-8"
            res = response.json()

            if "error_code" in res:
                error_code = res["error_code"]
                error_msg = res["error_msg"]
                error_hint = {
                    401: "API_KEY或SECRET_KEY错误（检查是否复制正确）",
                    403: "应用未开通文字识别接口（控制台→应用管理→勾选对应接口）",
                    17: "APP_ID错误（检查是否复制正确）",
                    18: "API_KEY不存在（确认应用已创建且API_KEY正确）",
                    110: "令牌过期或无效（重新运行程序获取新令牌）",
                    216200: "空图片验证（正常现象，仅用于测试凭证有效性）"
                }.get(error_code, "未知错误")
                print(f"❌ 凭证验证失败：{error_msg}（错误码：{error_code}）\n提示：{error_hint}")
                return False
            else:
                print("✅ 凭证验证成功，可正常调用接口")
                return True
        except Exception as e:
            print(f"❌ 凭证验证异常：{str(e)}（可能是网络问题或接口地址变更）")
            return False

    def _read_image(self, file_path: str) -> bytes:
        """读取图片并返回字节数据（用于后续裁剪）"""
        try:
            with open(file_path, 'rb') as f:
                return f.read()
        except Exception as e:
            print(f"❌ 图片读取失败：{str(e)}（检查路径/文件是否存在）")
            return b""

    def _crop_image(self, image_bytes: bytes, x1: int, y1: int, x2: int, y2: int) -> bytes:
        """裁剪图片指定区域，返回裁剪后的字节数据"""
        try:
            img = Image.open(io.BytesIO(image_bytes))
            cropped_img = img.crop((x1, y1, x2, y2))  # 裁剪矩形区域
            cropped_bytes = io.BytesIO()
            cropped_img.save(cropped_bytes, format="JPEG")  # 保存为JPEG
            return cropped_bytes.getvalue()
        except Exception as e:
            print(f"❌ 图片裁剪失败：{str(e)}")
            return b""

    def _get_material_type(self, material_name: str) -> str:
        """判断材料类型"""
        for type_name, keywords in self.material_type_map.items():
            if any(kw in material_name for kw in keywords):
                return type_name
        return "票据类"

    def _ocr_general_high_accuracy(self, image_base64: str) -> Dict:
        """通用高精度识别（票据类）"""
        if not image_base64:
            return {"error": "图片内容为空，无法识别"}
        if not self.access_token:
            return {"error": "未获取到接口访问令牌（重新运行程序重试）"}

        ocr_url = f"https://aip.baidubce.com/rest/2.0/ocr/v1/accurate?access_token={self.access_token}"
        data = {
            "image": image_base64,
            "detect_direction": "true",
            "probability": "true"
        }

        try:
            response = requests.post(
                ocr_url,
                data=data,
                headers={"Content-Type": "application/x-www-form-urlencoded; charset=utf-8"}
            )
            response.encoding = "utf-8"
            return response.json()
        except Exception as e:
            return {"error": f"票据识别接口调用失败：{str(e)}"}

    def _ocr_table_async(self, image_base64: str) -> Dict:
        """异步表格识别（报表类）"""
        if not image_base64:
            return {"error": "图片内容为空，无法识别"}
        if not self.access_token:
            return {"error": "未获取到接口访问令牌（重新运行程序重试）"}

        request_url = f"https://aip.baidubce.com/rest/2.0/ocr/v1/table_async?access_token={self.access_token}"
        data = {"image": image_base64}

        try:
            # 发送表格识别请求
            response = requests.post(
                request_url,
                data=data,
                headers={"Content-Type": "application/x-www-form-urlencoded; charset=utf-8"}
            )
            response.encoding = "utf-8"
            async_res = response.json()

            request_id = async_res.get("result", {}).get("request_id")
            if not request_id:
                return {"error": f"表格识别请求失败：{async_res.get('error_msg', '未知原因')}"}

            # 轮询获取结果（最多等待30秒）
            result_url = f"https://aip.baidubce.com/rest/2.0/ocr/v1/get_table_result?access_token={self.access_token}"
            for _ in range(15):
                res = requests.post(
                    result_url,
                    data={"request_id": request_id},
                    headers={"Content-Type": "application/x-www-form-urlencoded; charset=utf-8"}
                )
                res.encoding = "utf-8"
                res_json = res.json()
                if res_json.get("result", {}).get("status") == "Success":
                    return res_json
                time.sleep(2)
            return {"error": "表格识别超时（超过30秒）"}
        except Exception as e:
            return {"error": f"表格识别接口调用失败：{str(e)}"}

    def _parse_bill(self, ocr_result: Dict) -> Dict:
        """解析通用票据类信息"""
        if "error_code" in ocr_result:
            return {
                "material_type": "票据类",
                "error": f"{ocr_result.get('error_msg')}（错误码：{ocr_result.get('error_code')}）"
            }
        if "error" in ocr_result:
            return {"material_type": "票据类", "error": ocr_result["error"]}

        words_list = [item["words"].strip() for item in ocr_result.get("words_result", []) if item["words"].strip()]
        raw_text = "\n".join(words_list) if words_list else ""

        key_info = {
            "material_type": "票据类",
            "金额（大写）": "", "金额（小写）": "",
            "日期": "", "账号": "", "户名": "", "开户行": "",
            "raw_text": raw_text
        }

        # 提取金额（大写）
        capital_amount_start_idx = -1
        for idx, line in enumerate(words_list):
            if "人民币" in line or "（大写）" in line or "(大写)" in line:
                capital_amount_start_idx = idx
                break
        if capital_amount_start_idx != -1:
            capital_amount_lines = []
            for line in words_list[capital_amount_start_idx + 1:]:
                if any(key in line for key in ["日", "月", "年", "账号", "户名"]):
                    break
                if any(c in line for c in ["零", "壹", "贰", "叁", "肆", "伍", "陆", "柒", "捌", "玖", "拾", "佰", "仟", "万", "亿", "元", "角", "分", "整"]):
                    capital_amount_lines.append(line)
            key_info["金额（大写）"] = "".join(capital_amount_lines)

        # 提取金额（小写）
        lower_amount_start_idx = -1
        for idx, line in enumerate(words_list):
            if "￥" in line:
                lower_amount_start_idx = idx
                break
        if lower_amount_start_idx != -1:
            lower_amount = words_list[lower_amount_start_idx].replace("￥", "")
            for line in words_list[lower_amount_start_idx + 1:]:
                if all(c.isdigit() or c == "." for c in line):
                    lower_amount += line
                else:
                    break
            if "." not in lower_amount:
                lower_amount += ".00"
            key_info["金额（小写）"] = lower_amount

        # 提取日期
        date_start_idx = -1
        for idx, line in enumerate(words_list):
            if "日期" in line:
                date_start_idx = idx
                break
        if date_start_idx != -1:
            date_lines = []
            for line in words_list[date_start_idx + 1:]:
                if any(c in line for c in ["年", "月", "日"]):
                    date_lines.append(line)
                if len(date_lines) >= 3:
                    break
            key_info["日期"] = "".join(date_lines)

        # 提取账号、户名、开户行
        for line in words_list:
            if "账号" in line:
                key_info["账号"] = line.split("：")[-1].strip() if "：" in line else line.split("账号")[-1].strip()
                break
        for line in words_list:
            if any(key in line for key in ["户名", "收款人", "付款人"]):
                if "：" in line:
                    key_info["户名"] = line.split("：")[-1].strip()
                else:
                    for key in ["户名", "收款人", "付款人"]:
                        if key in line:
                            key_info["户名"] = line.split(key)[-1].strip()
                            break
            if key_info["户名"]:
                break
        for line in words_list:
            if "银行" in line and "人民币" not in line:
                key_info["开户行"] = line.strip()
                break
            if "开户行" in line:
                key_info["开户行"] = line.split("：")[-1].strip() if "：" in line else line.split("开户行")[-1].strip()
                break

        return key_info

    def _parse_table(self, ocr_result: Dict) -> Dict:
        """解析报表类表格信息"""
        if "error_code" in ocr_result:
            return {
                "material_type": "报表类",
                "error": f"{ocr_result.get('error_msg')}（错误码：{ocr_result.get('error_code')}）"
            }
        if "error" in ocr_result:
            return {"material_type": "报表类", "error": ocr_result["error"]}

        cells = ocr_result.get("result", {}).get("data", {}).get("cells", [])
        if not cells:
            return {
                "material_type": "报表类",
                "error": "未识别到表格内容",
                "table_data": []
            }

        sorted_cells = sorted(cells, key=lambda x: (x["row"], x["col"]))
        table_data = []
        current_row = 0
        current_row_cells = []

        for cell in sorted_cells:
            if cell["row"] != current_row:
                table_data.append(current_row_cells)
                current_row_cells = []
                current_row = cell["row"]
            current_row_cells.append(cell["word"].strip())
        table_data.append(current_row_cells)

        return {
            "material_type": "报表类",
            "table_data": table_data,
            "row_count": len(table_data),
            "col_count": max(len(row) for row in table_data) if table_data else 0
        }

    def recognize(self, image_path: str, material_name: str) -> Dict:
        """核心识别函数：进账单使用固定区域识别，其他材料使用通用识别"""
        image_bytes = self._read_image(image_path)
        if not image_bytes:
            return {
                "material_name": material_name,
                "material_type": "未知",
                "error": "图片读取失败，无法识别",
                "recognize_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            }

        material_type = self._get_material_type(material_name)
        print(f"\n=== 开始识别：{material_name} ===")
        print(f"=== 材料类型判定：{material_type} ===")

        try:
            # 进账单：使用固定区域分区识别
            if material_name == "进账单":
                result = {
                    "material_name": material_name,
                    "material_type": "票据类（进账单模板）",
                    "recognize_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                }
                # 遍历所有预定义区域，依次裁剪并识别
                for area_name, (x1, y1, x2, y2) in self.BILL_TEMPLATE_AREAS.items():
                    # 裁剪区域
                    cropped_bytes = self._crop_image(image_bytes, x1, y1, x2, y2)
                    if not cropped_bytes:
                        result[f"{area_name}_error"] = "区域裁剪失败"
                        continue
                    # 转换为Base64并调用OCR
                    cropped_base64 = base64.b64encode(cropped_bytes).decode("utf-8")
                    ocr_res = self._ocr_general_high_accuracy(cropped_base64)
                    # 提取区域文字（拼接所有识别行）
                    area_text = []
                    if "words_result" in ocr_res:
                        area_text = [item["words"].strip() for item in ocr_res["words_result"] if item["words"].strip()]
                    result[area_name] = " ".join(area_text) or "未识别到文字"
                return result

            # 其他票据：通用识别
            elif material_type == "票据类":
                image_base64 = base64.b64encode(image_bytes).decode("utf-8")
                ocr_result = self._ocr_general_high_accuracy(image_base64)
                print(f"=== 票据OCR原始结果：{ocr_result} ===")
                result = self._parse_bill(ocr_result)

            # 报表类：表格识别
            else:
                image_base64 = base64.b64encode(image_bytes).decode("utf-8")
                ocr_result = self._ocr_table_async(image_base64)
                print(f"=== 表格OCR原始结果：{ocr_result} ===")
                result = self._parse_table(ocr_result)

            # 补充基础信息
            result["material_name"] = material_name
            result["recognize_time"] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            return result

        except Exception as e:
            return {
                "material_name": material_name,
                "material_type": "未知",
                "error": f"识别失败：{str(e)}",
                "recognize_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            }

    def export_to_json(self, result: Dict, output_path: str):
        """将识别结果导出为JSON文件"""
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"JSON结果已导出至：{output_path}\n")


# 示例使用
if __name__ == "__main__":
    # 百度OCR凭证（从百度AI控制台获取：https://console.bce.baidu.com/ai/）
    APP_ID = "120177594"       # 替换为实际APP_ID
    API_KEY = "OQKBPivmg6iPvlcDECnM3msX"     # 替换为实际API_KEY
    SECRET_KEY = "NFIwcoy7VNCuFcymh22hOzmZiUktBqYd"  # 替换为实际SECRET_KEY

    # 初始化识别工具
    ocr_tool = LoanMaterialOCR(APP_ID, API_KEY, SECRET_KEY)

    # 待识别材料（进账单优先走分区识别）
    materials = [
        {
            "path": r"C:\\Users\\颖\\Desktop\\0BF9F651-FF22-C4F4-6503-8B1248C22EC4.jpg",  # 替换为实际图片路径
            "name": "进账单",  # 材料名称（触发进账单模板识别）
            "output": "进账单识别结果.json"
        }
    ]

    # 识别并导出结果
    for mat in materials:
        result = ocr_tool.recognize(mat["path"], mat["name"])
        print("识别结果预览：")
        if "票据类（进账单模板）" in result.get("material_type", ""):
            # 进账单预览所有区域结果
            for key in result:
                if key not in ["material_name", "material_type", "recognize_time"]:
                    print(f"{key}：{result[key]}")
        elif result.get("material_type") == "票据类":
            # 其他票据预览关键字段
            preview_fields = ["金额（大写）", "金额（小写）", "日期", "账号", "户名", "开户行", "error"]
            print({k: result.get(k, "") for k in preview_fields})
        else:
            # 报表类预览表格结构
            print(f"表格行数：{result.get('row_count')}，列数：{result.get('col_count')}")
            print("表格前2行：", result.get("table_data", [])[:2])
        # 导出JSON
        ocr_tool.export_to_json(result, mat["output"])