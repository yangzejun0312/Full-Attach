import json
import os
import random
import time
from pathlib import Path
import requests
import asyncio
import base64
import aiohttp
project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
import sys
sys.path.append(project_dir)
from utils.prompt_process import get_prompt


# import zhipuai
# zhipuai.api_key = "3fe121b978f1f456cfac1d2a1a9d8c06.iQsBvb1F54iFYfZq"
LLM_CONFIG_DIR = os.path.dirname(__file__)


def get_model_answer(model_name, inputs_list, user_dir=LLM_CONFIG_DIR, stream=False):
    # 读取LLM_CONFIG

    answer = 'no answer'
    if 'gpt' in model_name:
        model = OPENAI_API(model_name, user_dir=user_dir)
        answer = model.get_response(inputs_list, stream=stream)
    elif 'glm-4v-plus-0111' in model_name:  
       model = GLM4V_API(model_name, user_dir=user_dir)
       answer = model.get_response(inputs_list)
    elif 'glm-4-plus' in model_name:  
       model = GLM4_API(model_name, user_dir=user_dir)
       answer = model.get_response(inputs_list)
    else:
        model = OPENAI_API(model_name, user_dir=user_dir)  # 代理站中一般可以访问多种OPENAI接口形式的自定义模型，这里作为保底。
        answer = model.get_response(inputs_list, stream=stream)
    return answer

class GLM4V_API:
    """智谱视觉语言模型调用类"""
    def __init__(self, model_name, user_dir):
        self.USER_DIR = Path(user_dir)
        self.CONFIG_PATH = self.USER_DIR / "config"
        config = json.load(open(self.CONFIG_PATH / "lm_api.json"))
        self.config = config["GLM4V_CONFIG"]
        self.model_name = model_name
        
        self.api_key = self.config["API_KEY"]
        self.base_url = self.config.get("BASE_URL", "https://open.bigmodel.cn/api/paas/v4/chat/completions")
        self.temperature = self.config.get("TEMPERATURE")
        self.max_tokens = self.config.get("MAX_TOKENS", 2000)
        self.max_retries = self.config.get("MAX_RETRIES", 5)
        self.timeout = aiohttp.ClientTimeout(total=30)

    async def get_response_async(self, inputs_list):
        """异步调用实现"""
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }
        
        async with aiohttp.ClientSession(timeout=self.timeout) as session:
            for attempt in range(self.max_retries):
                try:
                    async with session.post(
                        self.base_url,
                        headers=headers,
                        json={
                            "model": self.model_name,
                            "messages": inputs_list,
                            "temperature": self.temperature,
                            "max_tokens": self.max_tokens,
                            "response_format": {"type": "json_object"}
                        }
                    ) as response:
                        if response.status == 200:
                            data = await response.json()
                            return data["choices"][0]["message"]["content"]
                        else:
                         error = await response.text()
                         raise Exception(f"API 请求失败: {error}")
                except Exception as e:
                    await asyncio.sleep(2 ** attempt)
                    
            raise Exception("Max retries exceeded")

    def get_response(self, inputs_list):
        """同步封装接口"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop.run_until_complete(self.get_response_async(inputs_list))
    
class GLM4_API:
    """智谱LLM模型调用类"""
    def __init__(self, model_name, user_dir):
        self.USER_DIR = Path(user_dir)
        self.CONFIG_PATH = self.USER_DIR / "config"
        config = json.load(open(self.CONFIG_PATH / "lm_api.json"))
        self.config = config["GLM4_CONFIG"]
        self.model_name = model_name
        
        self.api_key = self.config["API_KEY"]
        self.base_url = self.config.get("BASE_URL")
        self.temperature = self.config.get("TEMPERATURE")
        self.max_tokens = self.config.get("MAX_TOKENS")
        self.max_retries = self.config.get("MAX_RETRIES")
        self.timeout = aiohttp.ClientTimeout(total=30)

    async def get_response_async(self, inputs_list):
        """异步调用实现"""
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }
        
        async with aiohttp.ClientSession(timeout=self.timeout) as session:
            for attempt in range(self.max_retries):
                try:
                    async with session.post(
                        self.base_url,
                        headers=headers,
                        json={
                            "model": self.model_name,
                            "messages": inputs_list,
                            "temperature": self.temperature,
                            "max_tokens": self.max_tokens,
                            "response_format": {"type": "json_object"}
                        }
                    ) as response:
                        if response.status == 200:
                            data = await response.json()
                            return data["choices"][0]["message"]["content"]
                        else:
                         error = await response.text()
                         raise Exception(f"API 请求失败: {error}")
                except Exception as e:
                    await asyncio.sleep(2 ** attempt)
                    
            raise Exception("Max retries exceeded")

    def get_response(self, inputs_list):
        """同步封装接口"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop.run_until_complete(self.get_response_async(inputs_list))




class BAIDU_API:
    # DOC: https://cloud.baidu.com/doc/WENXINWORKSHOP/s/4lilb2lpf
    # 充值: https://console.bce.baidu.com/billing/#/account/index
    # 开通新模型: https://console.bce.baidu.com/qianfan/chargemanage/list

    def __init__(self):
        API_KEY = "qq7WLVgNX88unRoUVLtNz8fQ"  # qq7WLVgNX88unRoUVLtNz8fQ
        SECRET_KEY = "gA3VOdcRnGM4gKKkKKi93A79Dwevm3zo"  # gA3VOdcRnGM4gKKkKKi93A79Dwevm3zo
        self.access_token = None
        self.api_key = API_KEY
        self.secret_key = SECRET_KEY
        self.api_base = "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/eb-instant?access_token="
        self.get_access_token()

    def convert_openai_to_baidu(self, inputs_list):
        """
        将 OpenAI 的输入转换为百度的输入
        检测是否为偶数，如果为偶数，那就把system拼接到user上面

        :param inputs_list: OpenAI 的输入
        :return: 百度的输入
        """
        combined_content = '\n\n'.join(item['content'].strip() for item in inputs_list)
        baidu_inputs_list = [{"role": "user", "content": combined_content}]
        return baidu_inputs_list

    def get_response(self, inputs_list):
        self.url = self.api_base + self.access_token
        payload = json.dumps({
            "messages": self.convert_openai_to_baidu(inputs_list),
            "temperture": "0"
        })
        headers = {
            'Content-Type': 'application/json'
        }
        response = requests.request("POST", self.url, headers=headers, data=payload)
        # load json data
        data = json.loads(response.text)
        response = data["result"]
        return response

    def get_access_token(self):
        """
        使用 AK，SK 生成鉴权签名（Access Token）
        :return: access_token，或是None(如果错误)
        """
        url = "https://aip.baidubce.com/oauth/2.0/token"
        params = {"grant_type": "client_credentials", "client_id": self.api_key, "client_secret": self.secret_key}
        self.access_token = str(requests.post(url, params=params).json().get("access_token"))
        return self.access_token


class OPENAI_API:
    def __init__(self, model_name, user_dir):
        self.USER_DIR = Path(user_dir)
        self.CONFIG_PATH = self.USER_DIR / "config"
        # 读取LLM_CONFIG
        OPENAI_CONFIG_PATH = self.CONFIG_PATH / "llm_config_template.json"
        openai_config_data = json.load(open(OPENAI_CONFIG_PATH, "r"))
        self.keys_bases = openai_config_data["OPENAI_CONFIG"]["OPENAI_KEYS_BASES"]
        self.current_key_index = 0  # 初始索引
        self.api_key, self.api_base = self.keys_bases[self.current_key_index]["OPENAI_KEY"], \
            self.keys_bases[self.current_key_index]["OPENAI_BASE"]

        self.model_name = model_name
        self.max_tokens = openai_config_data["OPENAI_CONFIG"]["OPENAI_MAX_TOKENS"]
        self.temperature = openai_config_data["OPENAI_CONFIG"]["OPENAI_TEMPERATURE"]
        self.stop = None
        self.client = None
        self.load_model()

    def load_model(self):
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.api_base
        )

    def switch_api_key(self):
        self.current_key_index = (self.current_key_index + 1) % len(self.keys_bases)
        new_key_base = self.keys_bases[self.current_key_index]
        self.api_key, self.api_base = new_key_base["OPENAI_KEY"], new_key_base["OPENAI_BASE"]
        self.load_model()
        print(f"Switched to new API key and base: {self.api_key}, {self.api_base}")

    def get_response(self, inputs_list, stream=False, max_retries=3):
        attempt = 0
        while attempt < max_retries:
            try:
                if stream:
                    print("----- Streaming Request -----")
                    stream_response = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=inputs_list,
                        temperature=self.temperature,  # 对话系统需要启动随机性
                        stream=True,
                    )
                    return stream_response
                else:
                    response = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=inputs_list,
                        max_tokens=self.max_tokens,
                        temperature=self.temperature,
                        stop=self.stop
                    )
                    # print(response)
                    return response.choices[0].message.content
            except Exception as e:
                attempt += 1
                print(f"Attempt {attempt} failed with error: {e}")
                if attempt < max_retries:
                    wait_time = (2 ** attempt) + random.uniform(0, 1)  # Exponential backoff with jitter
                    print(f"Waiting {wait_time:.2f} seconds before retrying...")
                    time.sleep(wait_time)
                    self.switch_api_key()  # Optionally switch API key before retrying
                else:
                    return "An error occurred, and the request could not be completed after retries."


class GEMINI:
    def __init__(self, model_name, project_root_path):
        self.PROJECT_ROOT_PATH = Path(project_root_path)
        self.CONFIG_PATH = self.PROJECT_ROOT_PATH / "config"
        # 读取配置
        GEMINI_CONFIG_PATH = self.CONFIG_PATH / "llm_config_template.json"
        gemini_config_data = json.load(open(GEMINI_CONFIG_PATH, "r"))
        self.api_keys = gemini_config_data["GEMINI_CONFIG"]["GEMINI_KEYS"]
        self.api_usage_limit = gemini_config_data["GEMINI_CONFIG"].get("API_USAGE_LIMIT", 1000)
        self.api_usage = {key: 0 for key in self.api_keys}  # 初始化每个key的使用次数
        self.temperature = gemini_config_data["GEMINI_CONFIG"]["GEMINI_TEMPERATURE"]
        self.model_name = model_name
        self.model = genai.GenerativeModel(self.model_name)
        self.proxy = gemini_config_data["GEMINI_CONFIG"].get("PROXY", {"http": "", "https": ""})
        self.current_api_key_index = 0  # 初始索引
        self.configure_api(self.api_keys[self.current_api_key_index])
        self.max_tokens = gemini_config_data["GEMINI_CONFIG"]["GEMINI_MAX_TOKENS"]

    def configure_api(self, api_key):
        os.environ["HTTP_PROXY"] = self.proxy["http"]
        os.environ["HTTPS_PROXY"] = self.proxy["https"]
        genai.configure(api_key=api_key)

    def switch_api_key(self):
        self.current_api_key_index = (self.current_api_key_index + 1) % len(self.api_keys)
        self.configure_api(self.api_keys[self.current_api_key_index])
        print(f"Switched to new API key: {self.api_keys[self.current_api_key_index]}")

    def get_response(self, inputs_list):
        messages = []

        # 分别处理用户和模型的部分，确保不会添加空的内容到parts中
        user_parts = [input["content"] for input in inputs_list if
                      input["role"] in ["system", "user"] and input["content"]]
        if user_parts:  # 只有当有用户部分时才添加
            messages.append({'role': 'user', 'parts': user_parts})

        model_parts = [input["content"] for input in inputs_list if input["role"] == "assistant" and input["content"]]
        for part in model_parts:  # 对于模型的每一部分，分别添加
            messages.append({'role': 'model', 'parts': [part]})

        for retries in range(5):
            try:
                response = self.model.generate_content(messages, generation_config=genai.types.GenerationConfig(
                    temperature=self.temperature, max_output_tokens=self.max_tokens))
                answer = response.text

                # 更新API key使用次数并检查是否需要切换
                self.api_usage[self.api_keys[self.current_api_key_index]] += 1
                if self.api_usage[self.api_keys[self.current_api_key_index]] >= self.api_usage_limit:
                    self.switch_api_key()

                return answer
            except Exception as e:
                print(f"Error when calling the GEMINI API: {e}")
                if retries < 4:
                    print("Attempting to switch API key and retry...")
                    self.switch_api_key()
                else:
                    print("Maximum number of retries reached. The GEMINI API is not responding.")
                    return "I'm sorry, but I am unable to provide a response at this time due to technical difficulties."
                sleep_time = (2 ** retries) + random.random()
                print(f"Waiting for {sleep_time} seconds before retrying...")
                time.sleep(sleep_time)


class QwenLocal_API:
    def __init__(self, model_name, user_dir):
        self.USER_DIR = Path(user_dir)
        self.CONFIG_PATH = self.USER_DIR / "config"
        # 读取LLM_CONFIG
        config_data = json.load(open(self.CONFIG_PATH / "llm_config_template.json", "r"))
        self.server_name = config_data["QWEN_CONFIG"]["QWEN_SERVER_NAME"]
        self.port = config_data["QWEN_CONFIG"]["QWEN_SERVER_PORT"]
        self.max_tokens = config_data["QWEN_CONFIG"]["QWEN_MAX_TOKENS"]
        self.temperature = config_data["QWEN_CONFIG"]["QWEN_TEMPERATURE"]
        self.url = f"http://{self.server_name}:{self.port}/"

    def get_response(self, inputs_list, stream=False):
        payload = json.dumps({
            "messages": inputs_list,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature
        })
        headers = {
            'Content-Type': 'application/json'
        }
        if stream:
            # todo finish stream mode
            response = requests.request("POST", self.url, headers=headers, data=payload, stream=True)
            return response.iter_lines()
        else:
            response = requests.request("POST", self.url, headers=headers, data=payload)
            if response.status_code != 200:
                return f"Error: {response.status_code}"
            # load json data
            data = json.loads(response.text)
            response = data["response"]
        return response



if __name__ == '__main__':
    VLM_CONFIG_DIR = Path(os.path.abspath(__file__)).parents[1] 
 #测试输入的三视角图片   
    image_paths = [
    VLM_CONFIG_DIR / "data" / "input" / "image-hight1.png",
    VLM_CONFIG_DIR / "data" / "input" / "image-left1.png",
    VLM_CONFIG_DIR / "data" / "input" / "image-right1.png"
]
    
    base64_images = []
    for path in image_paths:
        with open(path, "rb") as f:
            base64_images.append(base64.b64encode(f.read()).decode('utf-8'))
    base64_img1, base64_img2, base64_img3 = base64_images
    # GLM-4V 多模态测试（需要先配置GLM4V_API的API_KEY）
    context_single = {
        0: {
            "action_history": ["left pick white part from table"],
            "action_result": ["success"],
            "knowledge": "篮子里可存放三个物体"
        }
    } 
    prompt = get_prompt("ObserveDecide",context_single)
    glm4v_test_case = [
        {
            "role": "system",
            # "content": "你是一名机器人专家，负责控制一台双臂机器人。机器人本体有两个机械臂，需要准确分析下一次执行装配动作"
             "content": prompt
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{base64_img1}", # 测试图片URL
                        "url": f"data:image/png;base64,{base64_img2}",
                        "url": f"data:image/png;base64,{base64_img3}"
                    }
                },
                {"type": "text", "text": "请分析以下工作场景图像,输出机械臂下一次执行装配动作"}
            ]
        }
    ]
    
    try:
        print("\n" + "="*40 + " GLM-4V测试 " + "="*40)
        glm_response = get_model_answer(
            model_name='glm-4v-plus-0111',
            inputs_list=glm4v_test_case,
            user_dir=VLM_CONFIG_DIR
        )
        print("API响应原始数据：\n", glm_response)
        
    except Exception as e:
        print(f"GLM-4V测试失败：{str(e)}")

