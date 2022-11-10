import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass
from multiprocessing import Process
from pathlib import Path
from typing import Generic

import numpy as np
import tqdm

from llazy.dataset import ChunkT


@dataclass
class DataGenerationTaskArg:
    process_idx: int  # is not pid
    number: int
    show_process_bar: bool
    base_path: Path
    extension: str = ".pkl"


class DataGenerationTask(ABC, Process, Generic[ChunkT]):
    arg: DataGenerationTaskArg

    def __init__(self, arg: DataGenerationTaskArg):
        self.arg = arg
        super().__init__()
        self.post_init_hook()

    @abstractmethod
    def post_init_hook(self) -> None:
        pass

    @abstractmethod
    def generate_single_data(self) -> ChunkT:
        pass

    def run(self) -> None:
        np.random.seed(self.arg.process_idx)
        disable_tqdm = not self.arg.show_process_bar

        for _ in tqdm.tqdm(range(self.arg.number), disable=disable_tqdm):
            chunk = self.generate_single_data()
            name = str(uuid.uuid4()) + self.arg.extension
            file_path = self.arg.base_path / name
            chunk.dump(file_path)
