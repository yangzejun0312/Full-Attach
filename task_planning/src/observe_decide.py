import os
import sys
project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_dir)
from src.history_process import HistoryProcess
from datetime import datetime
from utils.prompt_process import get_prompt
from utils.api_call import get_model_answer
from typing import List
import json
from basic_config.paths import *
from utils.log_utils import *

class ObserveDecide():
    #参数views-三视角图片list
    def __init__(self, instruction: str = "", vlm_name: str = "glm-4v-plus-0111",contexts: dict = None ,views: List[str] = None):
       # 如果 contexts 是 None，则初始化为空字典
        if contexts is None:
            contexts = {}
        # 如果 views 是 None，则初始化为空列表
        if views is None:
            views = []
        self.create_time = datetime.now().strftime("%Y-%m-%d")
        self.my_logger = High_Logger(log_file_name=f"{self.create_time}.log")

        self.instruction = instruction
        self.action_history = []
        self.action_result = []
        self.img_ref = set()
        self.progress = ""
        self.prompt_template = get_prompt("ObserveDecide",contexts)
        self.vlm_name = vlm_name
        # 初始化多帧三视角图片
        self.views = views  #默认每三秒传一次（现阶段）
    
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
    def call_vlm_api(self): #名称
        #vlm配置文件地址
        VLM_CONFIG_DIR = Path(os.path.abspath(__file__)).parents[1]  
        content = [
                {"type": "text", "text": self.instruction},
            ]      
        #执行历史、任务进展
        # history_content = HistoryProcess().get_content() 
        # content.append({
        #     "type": "text",
        #     "text": f"### 运行上下文\n{json.dumps(history_content, indent=2, ensure_ascii=False)}"
        # })
        for  view in self.views:
            content.extend([
                {"type": "image_url",
                 "image_url": {"url": f"data:image/png;base64,{view}"}}   #base64格式
            ])
        glm4v_test_case = [
        {
            "role": "system",
            "content": self.prompt_template
        },
        {
            "role": "user",
            "content": content
        }
       ]
        glm_response = get_model_answer(
               model_name='glm-4v-plus-0111',
               inputs_list=glm4v_test_case,
               user_dir=VLM_CONFIG_DIR
           )
        return glm_response   #返回vlm的生成策略