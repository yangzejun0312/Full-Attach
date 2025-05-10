from basic_config.paths import *
import jinja2
import json

PROMPT_PATH = TASK_PLANNING_ROOT / "config" / "sys_prompt_templates.py"

def get_prompt(module: str = "", contexts: dict = None) -> dict:
    # module: str = "HistoryProcess" or "ObserveDecide"
    
    with open(PROMPT_PATH, "r", encoding="utf-8") as f:
        prompts = json.load(f)
    messages = prompts.get(module, [])

    for index, context in contexts.items():
        if index < len(messages):
            template = jinja2.Template(messages[index])
            messages[index] = template.render(**context)
        else:
            raise IndexError(f"Index {index} out of range for messages.")
    
    return messages
