
import jinja2
import importlib.util
import sys
import os
from pathlib import Path
project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(project_dir)
from basic_config.paths import *

def get_prompt(module: str = "", contexts: dict = None) -> dict:
    # module: str = "HistoryProcess" or "ObserveDecide"
    
      # 动态导入模板文件
    PROMPT_PATH = Path(TASK_PLANNING_ROOT / "config" / "sys_prompt_templates.py")  # 根据实际路径调整
    
    # 动态导入模块
    spec = importlib.util.spec_from_file_location("prompt_templates", PROMPT_PATH)
    prompt_module = importlib.util.module_from_spec(spec)
    sys.modules["prompt_templates"] = prompt_module
    spec.loader.exec_module(prompt_module)
    
    # 获取模板内容
    messages = [prompt_module.prompt_templates.get(module, "")]
    
    # 渲染模板
    if contexts:
        for index, context in contexts.items():
            if index < len(messages):
                template = jinja2.Template(messages[index])
                messages[index] = template.render(**context)
            else:
                raise IndexError(f"Index {index} out of range for messages.")
    
    return messages[0]

if __name__ == "__main__":
       # 测试用例1：正常情况 - 单消息模板渲染
    print("=== 测试1: 正常渲染单消息模板 ===")
    context_single = {
        0: {
            "action_history": ["left pick white part from table","right pick white part from table"],
            "action_result": ["success","failure"],
            "progress": "物体已拾取完成" ,
            "knowledge": "篮子里可以放三个物体"
        }   
    }
    context_single2 = {
        0: {
            "action_history": ["left pick white part from table","right pick white part from table"],
            "action_result": ["success","failure"],
             "knowledge": "篮子里可以放三个物体",
            "style": {
                   "status": "success/failure",
                   "reason": "分析原因"
             },
            "style_example" : {
                "status": "failure",
                "reason": "苹果位置发生变化，导致拾取失败"
            }
            }
        }   
    try:
        result = get_prompt("HistoryProcess", context_single2)
        print(f"渲染结果:\n{result}")  
    except Exception as e:
        print(f"测试1失败: {str(e)}")

