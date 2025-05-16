import logging
import sys
import os
project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(project_dir)
from basic_config.paths import *

LOG_FILE_ROOT = TASK_PLANNING_ROOT / 'logs'

class High_Logger():
    def __init__(self, log_file_name: str = ""):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        self.log_file_path = LOG_FILE_ROOT / log_file_name

        self.set_logger()
    
    def set_logger(self):
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setLevel(logging.DEBUG)

        file_handler = logging.FileHandler(self.log_file_path, mode='a', encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)

        formatter = logging.Formatter(
            '[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        stream_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)

        self.logger.addHandler(stream_handler)
        self.logger.addHandler(file_handler)

    def record(self, level: str, message: str):
        if level == 'debug':
            self.logger.debug(message)
        elif level == 'info':
            self.logger.info(message)
        elif level == 'warning':
            self.logger.warning(message)
        elif level == 'error':
            self.logger.error(message)
        elif level == 'critical':
            self.logger.critical(message)
        else:
            raise ValueError(f"Invalid log level: {level}")