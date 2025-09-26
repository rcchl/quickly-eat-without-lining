#appid:120177594
#api key OQKBPivmg6iPvlcDECnM3msX
#secret key NFIwcoy7VNCuFcymh22hOzmZiUktBqYd
#AES Key：e5e36f38b971456f
import time
import json
import base64
import requests
from typing import Dict, List, Union


class LoanMaterialOCR:
    """贷款材料通用OCR识别工具（基于requests手动调用接口，支持多行关键信息提取）"""

    def __init__(self, app_id: str, api_key: str, secret_key: str):
        self.app_id = app_id
        self.api_key = api_key
        self.secret_key = secret_key
        self.access_token = ""  # 接口访问令牌
        self.material_type_map = {
            "票据类": ["支票", "进账单", "汇款单", "转账支票"],
            "报表类": ["利润表", "资产负债表", "现金流量表"]
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
            # 发送请求（强制UTF-8编码）
            response = requests.get(token_url, params=params,
                                    headers={"Content-Type": "application/json; charset=utf-8"})
            response.encoding = "utf-8"  # 明确指定响应编码
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
        # 通用文字识别接口（用于验证，空图片仅测试凭证，不识别内容）
        ocr_url = f"https://aip.baidubce.com/rest/2.0/ocr/v1/general_basic?access_token={self.access_token}"
        empty_image_base64 = base64.b64encode(b"").decode("utf-8")  # 空图片Base64
        data = {"image": empty_image_base64}

        try:
            # 发送POST请求（强制UTF-8编码）
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
                # 常见错误码解释
                error_hint = {
                    401: "API_KEY或SECRET_KEY错误（检查是否复制正确）",
                    403: "应用未开通文字识别接口（控制台→应用管理→勾选对应接口）",
                    17: "APP_ID错误（检查是否复制正确）",
                    18: "API_KEY不存在（确认应用已创建且API_KEY正确）",
                    110: "令牌过期或无效（重新运行程序获取新令牌）",
                    216200: "空图片验证（正常现象，仅用于测试凭证有效性，不影响实际识别）"
                }.get(error_code, "未知错误")
                print(f"❌ 凭证验证失败：{error_msg}（错误码：{error_code}）\n提示：{error_hint}")
                return False
            else:
                print("✅ 凭证验证成功，可正常调用接口")
                return True
        except Exception as e:
            print(f"❌ 凭证验证异常：{str(e)}（可能是网络问题或接口地址变更）")
            return False

    def _read_image(self, file_path: str) -> str:
        """读取图片并转为Base64字符串（百度OCR接口要求格式）"""
        try:
            with open(file_path, 'rb') as f:
                image_bytes = f.read()
                # 转为Base64并按UTF-8编码（关键：避免中文路径导致的编码问题）
                return base64.b64encode(image_bytes).decode("utf-8")
        except Exception as e:
            print(f"❌ 图片读取失败：{str(e)}（检查路径是否正确、文件是否存在）")
            return ""

    def _get_material_type(self, material_name: str) -> str:
        """判断材料类型"""
        for type_name, keywords in self.material_type_map.items():
            if any(kw in material_name for kw in keywords):
                return type_name
        return "票据类"

    def _ocr_general_high_accuracy(self, image_base64: str) -> Dict:
        """通用高精度识别（票据类，基于requests手动调用接口）"""
        if not image_base64:
            return {"error": "图片内容为空，无法识别"}
        if not self.access_token:
            return {"error": "未获取到接口访问令牌（重新运行程序重试）"}

        # 百度OCR通用高精度接口地址
        ocr_url = f"https://aip.baidubce.com/rest/2.0/ocr/v1/accurate?access_token={self.access_token}"
        # 请求参数（含中文也可正常编码，因为headers指定了UTF-8）
        data = {
            "image": image_base64,
            "detect_direction": "true",  # 检测文字方向
            "probability": "true"  # 返回识别置信度
        }

        try:
            # 发送POST请求（强制UTF-8编码，彻底解决latin-1问题）
            response = requests.post(
                ocr_url,
                data=data,
                headers={"Content-Type": "application/x-www-form-urlencoded; charset=utf-8"}
            )
            response.encoding = "utf-8"  # 确保响应中文正常显示
            return response.json()
        except Exception as e:
            return {"error": f"票据识别接口调用失败：{str(e)}（检查网络或令牌有效性）"}

    def _ocr_table_async(self, image_base64: str) -> Dict:
        """异步表格识别（报表类，基于requests手动调用接口）"""
        if not image_base64:
            return {"error": "图片内容为空，无法识别"}
        if not self.access_token:
            return {"error": "未获取到接口访问令牌（重新运行程序重试）"}

        # 百度OCR异步表格识别接口
        request_url = f"https://aip.baidubce.com/rest/2.0/ocr/v1/table_async?access_token={self.access_token}"
        data = {"image": image_base64}

        try:
            # 1. 发送表格识别请求
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

            # 2. 轮询获取结果（最多等待30秒，每2秒查一次）
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
            return {"error": "表格识别超时（超过30秒，可能图片过大）"}
        except Exception as e:
            return {"error": f"表格识别接口调用失败：{str(e)}"}

    def _parse_bill(self, ocr_result: Dict) -> Dict:
        """解析票据类为关键信息字典（优化：支持多行提取金额、日期）"""
        if "error_code" in ocr_result:
            return {
                "material_type": "票据类",
                "error": f"{ocr_result.get('error_msg')}（错误码：{ocr_result.get('error_code')}）"
            }
        if "error" in ocr_result:
            return {"material_type": "票据类", "error": ocr_result["error"]}

        # 1. 提取所有非空文字行，按识别顺序保存（用于多行匹配）
        words_list = []
        for item in ocr_result.get("words_result", []):
            word = item["words"].strip()
            if word:  # 过滤空行
                words_list.append(word)
        if not words_list:
            return {
                "material_type": "票据类",
                "error": "未识别到任何文字（可能图片模糊或无文字）",
                "raw_text": ""
            }
        raw_text = "\n".join(words_list)

        # 2. 初始化关键信息字典
        key_info = {
            "material_type": "票据类",
            "金额（大写）": "", "金额（小写）": "",
            "日期": "", "账号": "", "户名": "", "开户行": "",
            "raw_text": raw_text
        }

        # -------------------- 优化1：提取金额（大写）（支持跨行吗） --------------------
        # 逻辑：找到"人民币"或"(大写)"所在位置，取后续连续非符号行作为金额大写
        capital_amount_start_idx = -1
        # 触发关键词：包含"人民币" 或 "（大写）"（中文括号）或 "(大写)"（英文括号）
        for idx, line in enumerate(words_list):
            if "人民币" in line or "（大写）" in line or "(大写)" in line:
                capital_amount_start_idx = idx
                break
        # 若找到触发词，取后续行直到遇到非金额行（金额行通常不含"日""月""账号"等关键词）
        if capital_amount_start_idx != -1:
            capital_amount_lines = []
            # 从触发词下一行开始收集
            for line in words_list[capital_amount_start_idx + 1:]:
                # 排除非金额行（含以下关键词则停止）
                if any(key in line for key in ["日", "月", "年", "账号", "户名", "开户行", "用途"]):
                    break
                # 保留金额相关字符（中文数字、"元""角""分""整"）
                if any(c in line for c in
                       ["零", "壹", "贰", "叁", "肆", "伍", "陆", "柒", "捌", "玖", "拾", "佰", "仟", "万", "亿", "元",
                        "角", "分", "整"]):
                    capital_amount_lines.append(line)
            key_info["金额（大写）"] = "".join(capital_amount_lines)

        # -------------------- 优化2：提取金额（小写）（支持拼接多行数字） --------------------
        # 逻辑：找到"￥"所在行，拼接后续连续数字行（含"."）
        lower_amount_start_idx = -1
        for idx, line in enumerate(words_list):
            if "￥" in line:
                lower_amount_start_idx = idx
                break
        if lower_amount_start_idx != -1:
            lower_amount = words_list[lower_amount_start_idx].replace("￥", "")  # 先取"￥"后的内容
            # 拼接后续连续数字行（仅含数字和"."）
            for line in words_list[lower_amount_start_idx + 1:]:
                # 检查是否为纯数字/小数点行
                if all(c.isdigit() or c == "." for c in line):
                    lower_amount += line
                else:
                    break
            # 补充小数点（若只有整数，加".00"）
            if "." not in lower_amount:
                lower_amount += ".00"
            key_info["金额（小写）"] = lower_amount

        # -------------------- 优化3：提取日期（支持拼接"年/月/日"多行） --------------------
        # 逻辑：找到"日期"或"出票日期"触发词，拼接后续"年""月""日"行
        date_lines = []
        date_start_idx = -1
        # 触发关键词：包含"日期"或"出票日期"
        for idx, line in enumerate(words_list):
            if "日期" in line:
                date_start_idx = idx
                break
        if date_start_idx != -1:
            # 收集后续含"年""月""日"的行（通常连续3行：年、月、日）
            for line in words_list[date_start_idx + 1:]:
                if any(c in line for c in ["年", "月", "日"]):
                    date_lines.append(line)
                # 最多收集3行（年+月+日），避免多余内容
                if len(date_lines) >= 3:
                    break
            key_info["日期"] = "".join(date_lines)

        # -------------------- 原有逻辑保留（优化关键词匹配范围） --------------------
        # 账号：包含"账号"的行，取"账号"后的内容（支持中文/英文冒号）
        for line in words_list:
            if "账号" in line:
                # 分割符：支持"账号：" "账号:" "账号 "
                if "：" in line:
                    key_info["账号"] = line.split("：")[-1].strip()
                elif ":" in line:
                    key_info["账号"] = line.split(":")[-1].strip()
                else:
                    key_info["账号"] = line.split("账号")[-1].strip()
                break  # 找到第一个账号行即可（通常只有一个）

        # 户名：包含"户名""收款人""付款人"的行
        for line in words_list:
            if any(key in line for key in ["户名", "收款人", "付款人"]):
                if "：" in line:
                    key_info["户名"] = line.split("：")[-1].strip()
                elif ":" in line:
                    key_info["户名"] = line.split(":")[-1].strip()
                else:
                    # 若无冒号，取关键词后的内容（如"收款人 某某公司"）
                    for key in ["户名", "收款人", "付款人"]:
                        if key in line:
                            key_info["户名"] = line.split(key)[-1].strip()
                            break
            if key_info["户名"]:  # 找到后跳出循环
                break

        # 开户行：包含"开户行""银行"的行（排除"人民币"等干扰行）
        for line in words_list:
            if "银行" in line and "人民币" not in line:
                key_info["开户行"] = line.strip()
                break
            if "开户行" in line:
                if "：" in line:
                    key_info["开户行"] = line.split("：")[-1].strip()
                else:
                    key_info["开户行"] = line.split("开户行")[-1].strip()
                break

        return key_info

    def _parse_table(self, ocr_result: Dict) -> Dict:
        """解析报表类为表格结构化字典"""
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

        # 按行列排序表格单元格
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
        """核心识别函数"""
        # 读取图片并转为Base64
        image_base64 = self._read_image(image_path)
        if not image_base64:
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
            if material_type == "票据类":
                ocr_result = self._ocr_general_high_accuracy(image_base64)
                print(f"=== 票据OCR原始结果：{ocr_result} ===")  # 调试用
                result = self._parse_bill(ocr_result)
            else:
                ocr_result = self._ocr_table_async(image_base64)
                print(f"=== 表格OCR原始结果：{ocr_result} ===")  # 调试用
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


# 示例使用（仅识别指定图片）
if __name__ == "__main__":
    # 1. 百度OCR凭证（从百度AI控制台获取：https://console.bce.baidu.com/ai/）
    APP_ID = "120177594"  # 用户提供的APP_ID
    API_KEY = "OQKBPivmg6iPvlcDECnM3msX"  # 用户提供的API_KEY
    SECRET_KEY = "NFIwcoy7VNCuFcymh22hOzmZiUktBqYd"  # 用户提供的SECRET_KEY

    # 2. 初始化识别工具（自动获取令牌+验证凭证）
    ocr_tool = LoanMaterialOCR(APP_ID, API_KEY, SECRET_KEY)

    # 3. 待识别的材料（仅指定图片）
    materials = [
        {
            "path": r"C:\Users\颖\Desktop\0BF9F651-FF22-C4F4-6503-8B1248C22EC4.jpg",  # 替换为指定路径
            "name": "进账单",  # 材料名称（用于判定类型）
            "output": "receipt_result.json"  # 输出JSON文件名
        }
    ]

    # 4. 识别并导出结果
    for mat in materials:
        result = ocr_tool.recognize(mat["path"], mat["name"])
        print("识别结果预览：")
        if result.get("material_type") == "票据类":
            # 票据类预览关键字段
            preview_fields = ["material_name", "金额（大写）", "金额（小写）", "日期", "账号", "户名", "开户行", "error"]
            print({k: result.get(k, "") for k in preview_fields})
        else:
            # 报表类预览表格结构
            print(f"表格行数：{result.get('row_count')}，列数：{result.get('col_count')}")
            print("表格前2行：", result.get("table_data", [])[:2])
        # 导出JSON
        ocr_tool.export_to_json(result, mat["output"])