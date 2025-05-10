from task_planning.utils.log_utils import *
from datetime import datetime

class ObserveDecide():
    def __init__(self, instruction: str = ""):
        self.create_time = datetime.now().strftime("%Y-%m-%d")
        self.my_logger = High_Logger(log_file_name=f"{self.create_time}.log")

        self.instruction = instruction
        self.action_history = []
        self.action_result = []
        self.img_ref = set()
        self.progress = ""
    
    def add_action(self, action: str, result: str = ""):
        self.action_history.append(action)
        self.action_result.append(result)
        self.my_logger.info(f"Action added: {action}")

    def add_img_ref(self, img_ref: str):
        self.img_ref.add(img_ref)
        self.my_logger.info(f"Image reference added: {img_ref}")

    def summarize(self):
        pass

    def get_content(self, string: bool = True):
        if string:
            content = {
                "instruction": self.instruction,
                "action_history": "\n".join(self.action_history),
                "action_result": "\n".join(self.action_result),
                "img_ref": list(self.img_ref),
                "progress": self.progress
            }
        else:
            content = {
                "instruction": self.instruction,
                "action_history": self.action_history,
                "action_result": self.action_result,
                "img_ref": list(self.img_ref),
                "progress": self.progress
            }
        return content
    
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