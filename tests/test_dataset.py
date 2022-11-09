import hashlib
import os
import pickle
import subprocess
import tempfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Type, TypeVar

import numpy as np
import torch

from llazy.dataset import (
    LazyDecomplessDataLoader,
    LazyDecomplessDataset,
    TorchChunkProtocol,
    _zip_command,
)

ExampleChunkT = TypeVar("ExampleChunkT", bound="ExampleChunkBase")


@dataclass  # type: ignore
class ExampleChunkBase(TorchChunkProtocol):
    data: np.ndarray

    def __len__(self) -> int:
        return 1

    @classmethod
    def load(cls: Type[ExampleChunkT], path: Path) -> ExampleChunkT:
        with path.open(mode="rb") as f:
            data = pickle.load(f)
        return cls(data)

    def dump(self, path: Path) -> None:
        with path.open(mode="wb") as f:
            pickle.dump(self.data, f)


class ExampleChunk(ExampleChunkBase):
    def to_tensors(self) -> torch.Tensor:
        a = torch.from_numpy(self.data).float()
        return a


class ExampleChunk2(ExampleChunkBase):
    def to_tensors(self) -> Tuple[torch.Tensor, ...]:
        a = torch.from_numpy(self.data).float()
        return a, a


def test_dataset():
    def compute_hashint(chunk: ExampleChunk) -> int:
        byte = hashlib.md5(pickle.dumps(chunk)).digest()
        return int.from_bytes(byte, "big", signed=True)

    with tempfile.TemporaryDirectory() as td:
        base_path = Path(td)

        # prepare chunks
        hash_value_original = 0
        n_chunk = 200
        for _ in range(n_chunk):
            data = np.random.randn(56, 56, 28)
            chunk = ExampleChunk(data)
            name = str(uuid.uuid4()) + ".pkl"
            path = base_path / name
            chunk.dump(path)
            cmd = "{} -1 -f {}".format(_zip_command, path)
            subprocess.run(cmd, shell=True)

            hash_value_original += compute_hashint(chunk)

        # load
        elapsed_times = []
        for n_worker in [1, 2]:
            dataset = LazyDecomplessDataset.load(base_path, ExampleChunk, n_worker=n_worker)

            hash_value_load = 0

            ts = time.time()
            indices_per_chunks = np.array_split(np.array(range(n_chunk)), 3)
            for indices in indices_per_chunks:
                chunks = dataset.get_data(indices)
                assert len(chunks) == len(indices), "{} <-> {}".format(len(chunks), len(indices))

                for chunk in chunks:
                    hash_value_load += compute_hashint(chunk)
            elapsed_times.append(time.time() - ts)

            # compare hash
            assert hash_value_original == hash_value_load

        cpu_count = os.cpu_count()
        assert cpu_count is not None
        has_morethan_two_core = cpu_count >= 4
        if has_morethan_two_core:
            print(elapsed_times)
            assert elapsed_times[1] < elapsed_times[0] * 0.7


def test_dataloader():
    with tempfile.TemporaryDirectory() as td:
        base_path = Path(td)

        # prepare chunks
        n_chunk = 120
        for _ in range(n_chunk):
            data = np.random.randn(2, 3, 4)
            chunk = ExampleChunk(data)
            name = str(uuid.uuid4()) + ".pkl"
            path = base_path / name
            chunk.dump(path)
            cmd = "{} -1 -f {}".format(_zip_command, path)
            subprocess.run(cmd, shell=True)

        batch_size = 50

        for chunk_t in [ExampleChunk, ExampleChunk2]:
            dataset = LazyDecomplessDataset.load(base_path, chunk_t, n_worker=2)  # type: ignore
            loader = LazyDecomplessDataLoader(dataset, batch_size=batch_size)
            for sample in loader:
                if chunk_t == ExampleChunk2:
                    assert len(sample) == 2
                    sample = sample[0]
                assert isinstance(sample, torch.Tensor)
                assert sample.dim() == 4
                n_batch, a, b, c = sample.shape
                assert n_batch in (batch_size, n_chunk % batch_size)
                assert a == 2
                assert b == 3
                assert c == 4


if __name__ == "__main__":
    # test_dataset()
    test_dataloader()
