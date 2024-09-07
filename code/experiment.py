import datetime
import logging
import os
import random
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn
from typeguard import typechecked

from boundflow.network import Model

EXPERIMENT_OUTPUT_PATH = "../experiments/"


@typechecked
class Experiment:

    def __init__(self, args: Namespace):
        self.directory = self.__setup_experiment_directory(args.experiment)
        self.logger = self.__create_logger(self.directory)
        self.set_random_seed(args.seed)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger.info(args)
        self.logger.info(f"Running on {self.device}")

    @staticmethod
    def __setup_experiment_directory(sub_directory: str) -> Path:
        parent_dir = Path(EXPERIMENT_OUTPUT_PATH).absolute()
        experiment_dir = parent_dir / sub_directory
        experiment_dir.mkdir(parents=True, exist_ok=True)
        return experiment_dir

    @staticmethod
    def __create_logger(log_dir: Path) -> logging.Logger:
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        log_file = log_dir / f"logs_{datetime.datetime.now().isoformat(timespec='seconds')}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        return logger

    @staticmethod
    def set_random_seed(seed: int):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(seed)
        random.seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)

    def save_model(self, model: Model, identifier: str):
        model_path = self.__model_path(identifier)
        torch.save(model.state_dict(), model_path)

    def load_model(self, model: Model, identifier: str):
        model_path = self.__model_path(identifier)
        state_dict = torch.load(model_path, map_location="cpu")
        model.load_state_dict(state_dict)

    def __model_path(self, identifier: str):
        file_name = f"{identifier}.pt"
        return self.directory / file_name
