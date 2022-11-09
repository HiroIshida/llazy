import hashlib
import os
import pickle
import subprocess
import tempfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from llazy.dataset import ChunkBase, Dataset, _zip_command


@dataclass
class ExampleChunk(ChunkBase):
    data: np.ndarray

    def __len__(self) -> int:
        return 1

    @classmethod
    def load(cls, path: Path) -> "ExampleChunk":
        with path.open(mode="rb") as f:
            data = pickle.load(f)
        return cls(data)

    def dump(self, path: Path) -> None:
        with path.open(mode="wb") as f:
            pickle.dump(self.data, f)


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
            dataset = Dataset.load(base_path, ExampleChunk, n_worker=n_worker)

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


if __name__ == "__main__":
    test_dataset()
