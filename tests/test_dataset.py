import hashlib
import pickle
import subprocess
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from hugedata.dataset import ChunkBase, Dataset


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
        for _ in range(5):
            data = np.random.randn(50, 50, 50)
            chunk = ExampleChunk(data)
            name = str(uuid.uuid4()) + ".pkl"
            path = base_path / name
            chunk.dump(path)
            cmd = "gzip -1 -f {}".format(path)
            subprocess.run(cmd, shell=True)

            hash_value_original += compute_hashint(chunk)

        # load
        dataset = Dataset.load(base_path, ExampleChunk, n_worker=1)

        chunks1 = dataset.get_data(np.array([0, 1, 2]))
        assert len(chunks1) == 3
        chunks2 = dataset.get_data(np.array([3, 4]))
        assert len(chunks2) == 2

        hash_value_load = 0
        for chunk in chunks1 + chunks2:
            hash_value_load += compute_hashint(chunk)

        # compare hash
        assert hash_value_original == hash_value_load


if __name__ == "__main__":
    test_dataset()
