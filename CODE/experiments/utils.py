# experiments/utils.py
from __future__ import annotations

import os
import shutil
import time
from typing import Iterable


def rm_if_exists(path: str) -> None:
    if os.path.exists(path):
        os.remove(path)


def clean(paths: Iterable[str]) -> None:
    for p in paths:
        rm_if_exists(p)


def snapshot(db_path: str, snap_path: str) -> None:
    shutil.copy2(db_path, snap_path)


def rollback(snap_path: str, db_path: str) -> None:
    shutil.copy2(snap_path, db_path)


class Timer:
    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.t1 = time.perf_counter()

    @property
    def ms(self) -> float:
        return (self.t1 - self.t0) * 1000.0