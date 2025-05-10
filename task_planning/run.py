from datetime import datetime
import cv2
import logging
from utils.api import api_key
from zhipuai import ZhipuAI
import json
from concurrent.futures import ThreadPoolExecutor
import queue
import threading
import concurrent.futures
import asyncio
import aiohttp
from task_planning.config.sys_prompt_templates import  VLM_PROMPT_TEMPLATE
import os
from PIL import Image
import imagehash
from io import BytesIO
import base64
from typing import Dict, List, Optional
import numpy as np
from collections import OrderedDict
from datetime import datetime
import cv2

client = ZhipuAI(api_key=api_key)  # 填写自己的APIKey

# 日志配置
logging.basicConfig(
    level=logging.INFO,  # 设置日志级别（INFO 及以上会输出 WARNING）
    format='%(asctime)s - %(levelname)s - %(message)s',  # 包含时间、级别和消息
    handlers=[
        logging.FileHandler('logs/task_planning.log'),  # 输出到文件
        logging.StreamHandler()         # 输出到控制台
    ]
)


class TaskPlanning:
    def __init__(self, instruction=None, knowledge_file='data/input/knowledge.json'):
        """
        初始化任务规划器
        :param instruction: 自然语言指令
        :param knowledge_file: 静态知识库文件路径
        """
        self.instruction = instruction
        self.knowledge = self._load_knowledge(knowledge_file) if knowledge_file else {}
        self.history_data = self.load_reasoning_from_file("data/output/history.txt")
        self.llm_client = ZhipuAI(api_key=api_key)  # LLM客户端
        self.api_key = api_key
        self.cache = OrderedDict()  # 使用有序字典实现LRU缓存
        self.cache_size = 10  # 最大缓存条目数
        self.similarity_threshold = 8  # 哈希差异阈值
        self._last_instruction = None  # 跟踪指令变化

    def _load_knowledge(self, file_path: str) -> Dict:
        """加载静态知识库"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logging.error(f"知识库加载失败: {str(e)}")
            return {}

    async def __process_frame_set(self, frame_set: List[str], views: List[str]) -> Dict:
        """处理单组帧（三个视角）并生成指令，支持结果缓存"""
        try:
            # 1. 检查指令是否变化（清空缓存）
            if self.instruction != self._last_instruction:
                self.cache.clear()
                self._last_instruction = self.instruction
                logging.info("检测到指令变更，已清空缓存")

            # 2. 生成当前帧组的缓存键
            cache_key = self._get_cache_key(frame_set)

            # 3. 检查缓存命中
            cached_result = self._check_cache(cache_key)
            if cached_result:
                logging.info(f"缓存命中（哈希差异：{sum(cache_key[0] - k for k in cache_key)}）")
                return cached_result

            # 4. 未命中则调用VLM API
            logging.debug("未命中缓存，发起VLM API调用")
            vlm_result = await self._call_vlm_api(frame_set, views)

            # 5. 校验结果并更新缓存（仅当成功时）
            if "error" not in vlm_result:
                self._update_cache(cache_key, vlm_result)
                logging.debug(f"新增缓存条目（当前缓存大小：{len(self.cache)}/{self.cache_size}）")

            return vlm_result

        except Exception as e:
            error_msg = f"帧组处理异常: {str(e)}"
            logging.error(error_msg, exc_info=True)
            return {"error": error_msg}

    async def _call_vlm_api(self, frame_set: List[str], views: List[str]) -> Dict:
        """实际调用VLM API的逻辑"""
        vlm_prompt = VLM_PROMPT_TEMPLATE
        content = [
            {"type": "text", "text": "### 核心任务要求"},
            {"type": "text", "text": "1. 生成可直接执行的双臂控制指令\n2. 输出格式必须严格遵循下方要求"},
            {"type": "text", "text": "### 用户指令\n" + (self.instruction or "无明确指令")},
            {"type": "text", "text": "### 操作约束\n" + json.dumps({
                "allowed_actions": ["pick", "put", "pinch", "open", "stay"],
                "safety_rules": self.knowledge.get("safety_rules", [])
            }, indent=2)}
        ]

        for frame, view in zip(frame_set, views):
            content.extend([
                {"type": "text", "text": f"### {view}视角"},
                {"type": "image_url", "image_url": {"url": frame}}
            ])

        context = {
            "history": self.history_data["history"][-3:],
            "last_errors": self.history_data.get("errors", [])
        }
        content.append({
            "type": "text",
            "text": f"### 运行上下文\n{json.dumps(context, indent=2, ensure_ascii=False)}"
        })

        content.append({
            "type": "text",
            "text": "### 坐标系说明\n原点为机器人基座中心，单位米\n- X轴：正向朝前\n- Y轴：正向向左\n- Z轴：正向向上"
        })

        async with aiohttp.ClientSession() as session:
            async with session.post(
                    "https://open.bigmodel.cn/api/paas/v4/chat/completions",
                    headers={'Authorization': f'Bearer {self.api_key}', 'Content-Type': 'application/json'},
                    json={
                        "model": "glm-4v-plus-0111",
                        "messages": [
                            {"role": "system", "content": vlm_prompt["system"]},
                            {"role": "user", "content": content}
                        ],
                        "temperature": 0.3,
                        "response_format": {"type": "json_object"},
                        "max_tokens": 2000
                    }
            ) as response:
                if response.status != 200:
                    error_detail = await response.text()
                    logging.error(f"VLM API请求失败: {response.status}, 详情: {error_detail}")
                    return {"error": "API请求失败"}

                response_data = await response.json()
                return self._parse_response(response_data["choices"][0]["message"]["content"])
    def _parse_response(self, response_text: str) -> Dict:
        """解析并验证响应数据"""
        try:
            data = json.loads(response_text)

            # 基本结构校验
            required_keys = {"strategy", "to-vla_commands", "environment_analysis"}
            if not required_keys.issubset(data.keys()):
                missing = required_keys - data.keys()
                raise ValueError(f"响应缺少必要字段: {missing}")

            # 指令格式校验
            for cmd in data["to-vla_commands"]:
                # 校验基础字段
                if not all(k in cmd for k in ["arm", "action", "target", "position"]):
                    raise ValueError("指令缺少必要字段")

                # 校验坐标格式
                coord = cmd["position"]["coordinate"]
                if len(coord) != 3 or not all(isinstance(v, (int, float)) for v in coord):
                    raise ValueError("坐标格式错误，应为[x,y,z]")

                # 校验动作类型
                allowed_actions = ["pick", "put", "pinch", "open", "stay"]
                if cmd["action"] not in allowed_actions:
                    raise ValueError(f"非法动作类型: {cmd['action']}")

            return {
                "strategy": data["strategy"],
                "to-vla_commands": data["to-vla_commands"],
                "vlm_analysis": data["environment_analysis"]
            }
        except Exception as e:
            logging.error(f"响应解析失败: {str(e)}")
            return {"error": f"响应解析失败: {str(e)}"}

    def _parse_strategy(self, strategy_data: Dict) -> Dict:
        """解析策略数据"""
        # 可添加自定义校验逻辑
        return strategy_data

    async def __forward__(self, *video_paths: str) -> Dict:
        """
        异步改进版主流程：
        1. 多视频并行抽帧
        2. 动态帧组处理
        3. 集成决策结果分析
        """
        try:
            if len(video_paths) != 3:
                raise ValueError("需要且只能提供三个视角的视频路径")

            # 初始化控制变量
            self._stop_event = asyncio.Event()
            result = {
                "timestamp": datetime.utcnow().isoformat(),
                "processed_frames": 0,
                "last_decision": None,
                "decision_history": []
            }

            # 创建多视频帧队列
            frame_queues = [asyncio.Queue(maxsize=100) for _ in range(3)]
            views = ['顶视图', '左视图', '右视图']

            # 启动抽帧任务
            async def extract_frames(video_path, frame_queue, frames_per_second=0.5):
                cap = cv2.VideoCapture(video_path)
                try:
                    if not cap.isOpened():
                        logging.error(f"无法打开视频文件：{video_path}")
                        await frame_queue.put(None)  # 结束信号
                        return

                    fps = cap.get(cv2.CAP_PROP_FPS)
                    frame_interval = int(round(fps / frames_per_second))
                    frame_count = 0
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

                    while not self._stop_event.is_set():
                        ret, frame = cap.read()
                        if not ret:
                            break

                        if frame_count % frame_interval == 0:
                            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
                            frame_data = f"data:image/jpeg;base64,{base64.b64encode(buffer).decode('utf-8')}"
                            try:
                                await asyncio.wait_for(frame_queue.put(frame_data), timeout=1.0)
                            except asyncio.TimeoutError:
                                logging.warning(f"队列已满，跳过当前帧 - {video_path}")

                        if frame_count % 100 == 0:
                            logging.info(f"处理视频 {video_path} 的第 {frame_count} 帧，总帧数：{total_frames}")

                        frame_count += 1
                finally:
                    cap.release()
                    await asyncio.sleep(0)  # 允许事件循环切换

            # 启动异步抽帧任务
            tasks = [
                asyncio.create_task(extract_frames(video_paths[i], frame_queues[i]))
                for i in range(3)
            ]

            # 实时帧处理循环
            previous_frames = None  # 保存上一组帧用于结果分析
            while not self._stop_event.is_set():
                try:
                    current_frames = await asyncio.wait_for(
                        asyncio.gather(*[q.get() for q in frame_queues]),
                        timeout=5.0
                    )
                    if any(frame is None for frame in current_frames):  # 某个视频结束
                        raise StopIteration

                    logging.info(f"处理第 {result['processed_frames'] + 1} 帧组...")
                    try:
                        # 处理当前帧组生成决策
                        vlm_result = await self.__process_frame_set(current_frames, views)
                    except Exception as e:
                        logging.error(f"处理帧组时发生错误：{e}")
                        break

                    if vlm_result:
                        # 保存初始决策状态为pending
                        await self.__save_llm_decision(vlm_result, result="pending")
                        result["last_decision"] = vlm_result
                        result["processed_frames"] += 1

                        # 分析上一次决策结果（当有历史记录时）
                        if result["processed_frames"] > 1 and previous_frames:
                            try:
                                analysis_result = await self.analyze_decision_outcome(previous_frames)
                                result["decision_history"].append(analysis_result)
                                logging.info(f"决策结果分析: {analysis_result}")
                            except Exception as e:
                                logging.error(f"决策分析失败: {str(e)}")

                        # 更新历史帧数据
                        previous_frames = current_frames.copy()

                        # 紧急停止检查
                        if vlm_result.get("action") == "emergency_stop":
                            self._stop_event.set()
                            result["emergency_stop"] = True
                            for task in tasks:
                                task.cancel()
                            break

                except asyncio.TimeoutError:
                    logging.warning("帧获取超时，可能视频已结束")
                    break
                except StopIteration:
                    logging.info("至少一个视频处理完成")
                    break

            # 最终分析最后一次决策
            if previous_frames and result["processed_frames"] > 0:
                try:
                    analysis_result = await self.analyze_decision_outcome(previous_frames)
                    result["decision_history"].append(analysis_result)
                except Exception as e:
                    logging.error(f"最终决策分析失败: {str(e)}")

            # 等待所有抽帧任务完成
            await asyncio.gather(*tasks, return_exceptions=True)

            return result

        except Exception as e:
            error_result = {
                "error": str(e),
                "decision": "emergency_stop",
                "reason": "系统处理异常",
                "result": "failure"
            }
            try:
                await self.__save_llm_decision(error_result, result="failure")
            except Exception as save_error:
                logging.error(f"保存错误决策时发生错误：{save_error}")
            return error_result
        finally:
            # 确保所有任务被取消
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            # 显式关闭队列
            for q in frame_queues:
                await q.put(None)

    def __async_extract_frames(self, video_path: str, output_queue: queue.Queue, frames_per_second: float):
        """专用异步抽帧方法"""
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                logging.error(f"无法打开视频文件：{video_path}")
                output_queue.put(None)  # 结束信号
                return

            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_interval = int(round(fps / frames_per_second))
            frame_count = 0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            while not self._stop_event.is_set():
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_count % frame_interval == 0:
                    _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
                    frame_data = f"data:image/jpeg;base64,{base64.b64encode(buffer).decode('utf-8')}"
                    try:
                        output_queue.put(frame_data, timeout=1.0)  # 设置超时时间为1秒
                    except queue.Full:
                        logging.warning(f"队列已满，跳过当前帧 - {video_path}")

                if frame_count % 100 == 0:
                    logging.info(f"处理视频 {video_path} 的第 {frame_count} 帧，总帧数：{total_frames}")

                frame_count += 1

        except Exception as e:
            logging.error(f"抽帧异常: {video_path} - {str(e)}")
        finally:
            if 'cap' in locals():
                cap.release()
            output_queue.put(None)  # 结束信号

    async def __save_llm_decision(self, decision: Dict, result: str = "pending"):
        """保存VLM决策到历史文件"""
        history_file = "data/output/history.txt"
        os.makedirs(os.path.dirname(history_file), exist_ok=True)
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        new_record = {
            "timestamp": current_time,
            "decision": decision,
            "result": result
        }

        try:
            # 读取现有记录
            existing_records = []
            try:
                with open(history_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            existing_records.append(json.loads(line))
            except FileNotFoundError:
                pass

            # 检查是否存在相同决策的记录
            updated = False
            for record in existing_records:
                if record.get("decision") == new_record["decision"]:
                    record["timestamp"] = current_time
                    updated = True
                    break

            # 若未更新，则添加新记录
            if not updated:
                existing_records.append(new_record)

            # 重新写入所有记录（覆盖模式）
            with open(history_file, 'w', encoding='utf-8') as f:
                for record in existing_records:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")

        except Exception as e:
            logging.error(f"保存历史记录失败: {str(e)}")

    def _parse_response(self, raw_response: str) -> Dict:
        """解析模型原始响应"""
        try:
            if '```json' in raw_response:
                json_str = raw_response.split('```json')[1].split('```')[0]
            else:
                json_str = raw_response[raw_response.find('{'):raw_response.rfind('}') + 1]
            return json.loads(json_str)
        except json.JSONDecodeError:
            return {"error": "响应解析失败", "raw_response": raw_response}

    def save_reasoning_to_file(self, reasoning: str, filename: str):
        """保存推理结果"""
        try:
            with open(filename, 'a', encoding='utf-8') as f:
                f.write(f"{reasoning}\n---\n")
        except Exception as e:
            logging.error(f"保存失败: {str(e)}")

    def load_reasoning_from_file(self, filename: str = "data/output/history.txt") -> dict:
        """从历史文件加载推理记录，确保每条记录为字典"""
        history = {"history": []}
        try:
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            with open(filename, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)  # 直接解析每行为JSON
                        history["history"].append(record)
                    except json.JSONDecodeError as e:
                        logging.error(f"解析JSON失败: {line}，错误: {e}")
            logging.info(f"成功从 {filename} 加载 {len(history['history'])} 条记录")
        except FileNotFoundError:
            logging.error(f"文件 {filename} 不存在，将返回空历史记录")
        except Exception as e:
            logging.error(f"加载失败: {str(e)}")
        return history

    def _encode_image(self, image_path: str) -> str:
        """Base64编码单张图片"""
        with open(image_path, 'rb') as f:
            return f"data:image/jpeg;base64,{base64.b64encode(f.read()).decode()}"


    async def analyze_decision_outcome(self, current_frames: List[str]) -> Dict:
        """
        分析当前场景，评估历史决策的执行结果
        :param current_frames: 当前三个视角的帧base64数据
        :return: 包含分析结果和更新历史的响应
        """
        # 1. 调用VLM分析当前场景
        analysis_prompt = """
            你是一个机器人任务审计系统，需要完成以下任务：
            1. 分析当前三视角图像中的物体状态
            2. 对比最近一次决策的预期目标（参考history.txt）
            3. 判断该决策是否成功完成目标（success/failure）
            输出格式：{"status": "success/failure", "reason": "分析原因"}
        """
        vlm_response = await self.__call_vlm_api(current_frames, analysis_prompt)

        # 2. 解析响应并更新历史记录
        if vlm_response.get("status"):
            await self.__update_history_result(vlm_response["status"], vlm_response["reason"])
        if vlm_response.get("status") not in ["success", "failure"]:
            raise ValueError(f"无效状态值: {vlm_response.get('status')}")
        return vlm_response


    async def __call_vlm_api(self, frames: List[str], prompt: str) -> Dict:
        """调用VLM API进行场景分析"""
        content = [
            {"type": "text", "text": prompt},
            *[{"type": "image_url", "image_url": {"url": frame}} for frame in frames]
        ]
        async with aiohttp.ClientSession() as session:
            response = await session.post(
                "https://open.bigmodel.cn/api/paas/v4/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={
                    "model": "glm-4v-plus-0111",
                    "messages": [{"role": "user", "content": content}],
                    "response_format": {"type": "json_object"}
                }
            )
            data = await response.json()
            return json.loads(data["choices"][0]["message"]["content"])


    async def __update_history_result(self, status: str, reason: str):
        """更新最近一条历史记录的结果"""
        history = self.load_reasoning_from_file()
        if history["history"]:
            latest_record = history["history"][-1]
            latest_record["result"] = status
            latest_record["analysis"] = reason
            # 回写更新后的历史记录
            with open("data/output/history.txt", "w", encoding="utf-8") as f:
                for record in history["history"]:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
    def _compute_frame_hash(self, frame_data: str) -> imagehash.ImageHash:
        """计算单帧的感知哈希"""
        try:
            # 提取Base64数据并解码
            encoded_data = frame_data.split(",")[1]
            binary_data = base64.b64decode(encoded_data)
            image = Image.open(BytesIO(binary_data)).convert("L").resize((32, 32))  # 灰度+缩放加速计算
            return imagehash.phash(image)
        except Exception as e:
            logging.error(f"哈希计算失败: {str(e)}")
            return imagehash.ImageHash(np.zeros(256))  # 返回空哈希

    def _get_cache_key(self, frame_set: List[str]) -> tuple:
        """生成帧组的缓存键（三个视角哈希的元组）"""
        return tuple(self._compute_frame_hash(frame) for frame in frame_set)

    def _check_cache(self, current_key: tuple) -> Optional[Dict]:
        """检查缓存中是否存在相似帧组"""
        for cached_key in list(self.cache.keys()):
            # 计算总哈希差异
            total_diff = sum(h1 - h2 for h1, h2 in zip(current_key, cached_key))
            if total_diff <= self.similarity_threshold:
                # 移动条目到字典末尾表示最近使用（LRU）
                value = self.cache.pop(cached_key)
                self.cache[cached_key] = value
                return value
        return None

    def _update_cache(self, key: tuple, value: Dict):
        """更新缓存（LRU策略）"""
        if key in self.cache:
            self.cache.move_to_end(key)
        else:
            self.cache[key] = value
            # 超出容量时移除最旧条目
            if len(self.cache) > self.cache_size:
                self.cache.popitem(last=False)