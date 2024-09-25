from torch.utils.tensorboard import SummaryWriter
import logging
import os
import torch
import datetime
from configuration import config

class SummaryLogger:
    log_dir = None
    checkpoint_dir = None

    logger = None
    tensorboard = None

    @classmethod
    def init(cls):
        cls._generate_dir()
        cls.logger = logging.getLogger()
        cls.logger.setLevel(level=logging.DEBUG)

        file_handler = logging.FileHandler(os.path.join(cls.log_dir, "log.txt"))
        console_handle = logging.StreamHandler()
        file_handler.setLevel(logging.DEBUG)
        console_handle.setLevel(logging.DEBUG)
        cls.logger.addHandler(file_handler)
        cls.logger.addHandler(console_handle)

        cls.tensorboard = SummaryWriter(cls.log_dir)
        cls

    @classmethod
    def _generate_dir(cls):
        nowTime=datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
        cls.log_dir = os.path.join(config.root_path, "trainLogs", nowTime)
        if not os.path.exists(cls.log_dir):
            os.makedirs(cls.log_dir)
        
        cls.checkpoint_dir = os.path.join(config.root_path, "trainLogs", nowTime, "checkpoints")
        if not os.path.exists(cls.checkpoint_dir):
            os.makedirs(cls.checkpoint_dir)        
        return

    @classmethod
    def close(cls):
        cls.tensorboard.close()
        return

    @classmethod
    def save_model(cls, model, file_name):
        torch.save(model.state_dict(), os.path.join(cls.checkpoint_dir, file_name))
        return
    
    @classmethod
    def load_model(cls, model, file_path):
        model_weights = torch.load(file_path)
        model.load_state_dict(model_weights)
        return