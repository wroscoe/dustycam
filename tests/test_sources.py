import numpy as np
import cv2

from dustycam.nodes.sources import FileSource, create_source


def _write_dummy_images(tmp_path):
    images = []
    for i in range(2):
        img = np.full((8, 8, 3), fill_value=i * 50, dtype=np.uint8)
        path = tmp_path / f"img{i}.jpg"
        cv2.imwrite(str(path), img)
        images.append(path)
    return images


def test_file_source_reads_images(tmp_path):
    _write_dummy_images(tmp_path)
    src = FileSource(directory=tmp_path)

    frames = []
    while True:
        pkt = src.next_packet()
        if pkt is None:
            break
        frames.append(pkt)

    assert len(frames) == 2
    assert [pkt.frame_id for pkt in frames] == [0, 1]
    assert frames[0].image.shape == (8, 8, 3)


def test_create_source_prefers_file(tmp_path):
    _write_dummy_images(tmp_path)
    src = create_source(preferred="file", file_dir=tmp_path)
    assert isinstance(src, FileSource)
