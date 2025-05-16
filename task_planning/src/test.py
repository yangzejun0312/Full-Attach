# src/test.py
from observe_decide import ObserveDecide
import sys
import os
project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_dir)
from utils.prompt_process import get_prompt
from basic_config.paths import Path
import base64
import os
import json

if __name__ == "__main__":
    context_single = {
        0: {
            "action_history": ["left pick white part from table"],
            "action_result": ["success"],
            "knowledge": "篮子里可存放三个物体"
        }
    }
    VLM_CONFIG_DIR = Path(os.path.abspath(__file__)).parents[1]
    image_paths = [
    VLM_CONFIG_DIR / "data" / "input" / "image-hight1.png",
    VLM_CONFIG_DIR / "data" / "input" / "image-left1.png",
    VLM_CONFIG_DIR / "data" / "input" / "image-right1.png"
    ]

    # 初始化一个列表用于存储 base64 编码后的图片数据
    example_views = []
    
    # 遍历图片路径，读取并编码为 base64
    for path in image_paths:
        with open(path, "rb") as f:
            base64_img = base64.b64encode(f.read()).decode('utf-8')
        example_views.append(base64_img)

    task = ObserveDecide(
        instruction="请分析以下工作场景图像,输出机械臂下一次执行装配动作",
        vlm_name="glm-4v-plus-0111",
        contexts=context_single,
        views=example_views
    )
    try:
        print("\n正在调用VLM接口获取决策建议...")
        vlm_response = task.call_vlm_api()
        print("\nVLM返回结果：")
        if isinstance(vlm_response, dict):
            print(json.dumps(vlm_response, indent=2, ensure_ascii=False))
        else:
            print(vlm_response)
    except Exception as e:
        print(f"调用VLM接口时出错：{str(e)}")
        task.my_logger.record("ERROR", f"调用VLM接口时出错：{str(e)}")