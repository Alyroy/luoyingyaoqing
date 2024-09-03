# -*- coding:utf-8 -*-
import logging
from abc import ABC, abstractmethod

# LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
# logging.basicConfig(filename='logger.log', level=logging.DEBUG, format=LOG_FORMAT)

# logger = logging.getLogger(__name__)

class BaseModelEval(ABC):
    """
    评估基类：
    @read_prompt:prompt加载
    @main_eval:评估主函数体
    @result_parse:评估结果解析
    """
    def __init__(self, task_name):
        self.task_name = task_name
        # self.logger = logger
        # self.logger.info("Evaluation task:{}".format(task_name))

    @abstractmethod
    def read_prompt(self, *args, **kwargs):
        ...

    @abstractmethod
    def concat_prompt(self, *args, **kwargs):
        ...

    @abstractmethod
    def main_eval(self, *args, **kwargs):
        ...

    @abstractmethod
    def result_parse(self, *args, **kwargs):
        ...