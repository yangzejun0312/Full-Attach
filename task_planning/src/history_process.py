from task_planning.utils.log_utils import *
from datetime import datetime

class HistoryProcess():
    def __init__(self, instruction: str = "", llm_model: str = "gpt-3.5-turbo"):
        self.create_time = datetime.now().strftime("%Y-%m-%d")
        self.my_logger = High_Logger(log_file_name=f"{self.create_time}.log")

        self.instruction = instruction
        self.action_history = []
        self.action_result = [] # <img_1>
        # self.img_ref = dict() # <img_1>: path of the image 1
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