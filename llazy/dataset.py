import multiprocessing
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
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
        if self.n_worker == 1:
            paths = [self.compressed_path_list[i] for i in indices]
            return self.load_chunks(paths, self.chunk_type)
        else:
            assert False

    @staticmethod
    def load_chunks(paths: List[Path], chunk_type: Type[ChunkT]) -> List[ChunkT]:
        chunk_list: List[ChunkT] = []
        for path in paths:
            command = "gunzip --keep -f {}".format(path)
            subprocess.run(command, shell=True)
            path_decompressed = path.parent / path.stem
            chunk = chunk_type.load(path_decompressed)
            chunk_list.append(chunk)
        return chunk_list

    def _get_data_paralllel(self, indices: np.ndarray) -> List[ChunkT]:
        pool = multiprocessing.Pool(self.n_worker)
        np.array_split(indices)
        pool.map(data_generation_task, zip(range(n_cpu), n_process_list_assign))
        return self._get_data(indices)
