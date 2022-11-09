import multiprocessing
import os
import subprocess
from dataclasses import dataclass
from multiprocessing import Process
from pathlib import Path
from queue import Queue
from typing import Generic, List, Type, TypeVar

import numpy as np

ChunkT = TypeVar("ChunkT", bound="ChunkBase")


class ChunkBase:
    @classmethod
    def load(cls: Type[ChunkT], path: Path) -> ChunkT:
        ...

    def dump(self, path: Path) -> None:
        ...

    def __len__(self) -> int:
        ...


@dataclass
class Dataset(Generic[ChunkT]):
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
    ) -> "Dataset[ChunkT]":
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
            command = "gunzip --keep -f {}".format(path)
            subprocess.run(command, shell=True)
            path_decompressed = path.parent / path.stem
            chunk = chunk_type.load(path_decompressed)
            q.put(chunk)
