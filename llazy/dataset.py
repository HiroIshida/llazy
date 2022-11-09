import multiprocessing
import os
import subprocess
from abc import ABC, abstractmethod
from dataclasses import dataclass, fields
from multiprocessing import Process
from pathlib import Path
from queue import Queue
from typing import Generic, List, Optional, Tuple, Type, TypeVar, Union

import dill
import numpy as np
import torch
from torch.utils.data import default_collate

_has_gzip = subprocess.run("which gzip > /dev/null", shell=True).returncode == 0
_has_pigz = subprocess.run("which pigz > /dev/null", shell=True).returncode == 0
assert _has_gzip or _has_pigz
_unzip_command = "unpigz" if _has_pigz else "gunzip"
_zip_command = "pigz" if _has_pigz else "gzip"

ChunkT = TypeVar("ChunkT", bound="ChunkBase")
DillableChunkT = TypeVar("DillableChunkT", bound="DillableChunkBase")


class ChunkBase(ABC):
    @classmethod
    @abstractmethod
    def load(cls: Type[ChunkT], path: Path) -> ChunkT:
        pass

    @abstractmethod
    def dump_impl(self, path: Path) -> None:
        pass

    @abstractmethod
    def __len__(self) -> int:
        pass

    @abstractmethod
    def to_tensors(self) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        pass

    def dump(self, path: Path) -> None:
        self.dump_impl(path)
        command = "{0} -1 {1}".format(_zip_command, path)
        subprocess.run(command, shell=True)


@dataclass  # type: ignore
class DillableChunkBase(ChunkBase):
    @classmethod
    def load(cls: Type[DillableChunkT], path: Path) -> DillableChunkT:
        with path.open(mode="rb") as f:
            dic = dill.load(f)
        return cls(**dic)

    def dump_impl(self, path: Path) -> None:
        # as a shallow dict instead of dataclasses.asdict
        dic = {}
        for field in fields(self):
            key = field.name
            dic[key] = self.__dict__[key]
        with path.open(mode="wb") as f:
            dill.dump(dic, f)


@dataclass
class LazyDecomplessDataset(Generic[ChunkT]):
    base_path: Path
    compressed_path_list: List[Path]
    chunk_type: Type[ChunkT]
    n_worker: int
    parallelize_threshold: int = 20

    def __post_init__(self):
        if self.n_worker == -1:
            n_cpu = os.cpu_count()
            assert n_cpu is not None
            self.n_worker = n_cpu

    @classmethod
    def load(
        cls, base_path: Path, chunk_type: Type[ChunkT], n_worker: int = -1
    ) -> "LazyDecomplessDataset[ChunkT]":
        path_list = []
        for p in base_path.iterdir():
            if p.name.endswith(".gz"):
                path_list.append(p)
        return cls(base_path, path_list, chunk_type, n_worker)

    def __len__(self) -> int:
        return len(self.compressed_path_list)

    def get_data(self, indices: np.ndarray) -> List[ChunkT]:
        is_worth_parallelizing = len(indices) > self.parallelize_threshold
        q: "Queue[ChunkT]"
        if self.n_worker > 1 and is_worth_parallelizing:
            q = multiprocessing.Queue()
            indices_list_per_worker = np.array_split(indices, self.n_worker)
            process_list = []
            for indices_part in indices_list_per_worker:
                paths = [self.compressed_path_list[i] for i in indices_part]
                p = Process(target=self.load_chunks, args=(paths, self.chunk_type, q))
                p.start()
                process_list.append(p)

            chunk_list = [q.get() for _ in range(len(indices))]

            for p in process_list:
                p.join()

            return chunk_list
        else:
            q = Queue()
            paths = [self.compressed_path_list[i] for i in indices]
            self.load_chunks(paths, self.chunk_type, q)
            return list(q.queue)

    @staticmethod
    def load_chunks(paths: List[Path], chunk_type: Type[ChunkT], q: "Queue[ChunkT]") -> None:
        for path in paths:
            command = "{} --keep -f {}".format(_unzip_command, path)
            subprocess.run(command, shell=True)
            path_decompressed = path.parent / path.stem
            chunk = chunk_type.load(path_decompressed)
            q.put(chunk)


@dataclass
class LazyDecomplessDataLoader(Generic[ChunkT]):
    r"""
    A dataloader class that has the similar inteface as
    pytorch DataLoader
    """
    dataset: LazyDecomplessDataset[ChunkT]
    batch_size: int = 1
    shuffle: bool = True
    _indices_per_iter: Optional[List[np.ndarray]] = None  # set when __iter__ called

    def __iter__(self) -> "LazyDecomplessDataLoader[ChunkT]":
        n_dataset = len(self.dataset)
        indices = np.arange(n_dataset)
        if self.shuffle:
            np.random.shuffle(indices)

        (n_dataset // self.batch_size)

        indices_list_per_iter = []
        head = 0
        for i in range(n_dataset // self.batch_size):
            indices = np.arange(head, head + self.batch_size)
            indices_list_per_iter.append(indices)
            head += self.batch_size

        rem = n_dataset % self.batch_size
        if rem > 0:
            indices = np.arange(head, head + rem)
            indices_list_per_iter.append(indices)

        self._indices_per_iter = indices_list_per_iter
        return self

    def __next__(self) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        assert self._indices_per_iter is not None
        if len(self._indices_per_iter) == 0:
            raise StopIteration()

        indices = self._indices_per_iter.pop()
        chunk_list = self.dataset.get_data(indices)

        zipped = [chunk.to_tensors() for chunk in chunk_list]
        return default_collate(zipped)
