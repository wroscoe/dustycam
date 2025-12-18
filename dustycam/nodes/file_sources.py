from __future__ import annotations

import time
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

class FileSource:
    """Reads images from a directory in sorted order."""

    def __init__(self, directory: Path | str, loop: bool = False):
        self.directory = Path(directory)
        if not self.directory.exists():
            raise FileNotFoundError(f"Directory does not exist: {self.directory}")
        self.loop = loop
        self._files = self._list_images(self.directory)
        self._idx = 0

    def _list_images(self, directory: Path) -> List[Path]:
        patterns = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
        files: List[Path] = []
        for pattern in patterns:
            files.extend(directory.glob(pattern))
        files = sorted(files)
        if not files:
            raise FileNotFoundError(f"No images found in {directory}")
        return files

    def read(self) -> Optional[np.ndarray]:
        if self._idx >= len(self._files):
            if not self.loop:
                return None
            self._idx = 0
        path = self._files[self._idx]
        self._idx += 1
        frame = cv2.imread(str(path))
        if frame is None:
            # Skip unreadable files
            return self.read()
        return frame

    def close(self):
        pass