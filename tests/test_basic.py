#!/usr/bin/env python
# -*- coding: utf-8 -*-
from pathlib import Path
from time import time

import numpy as np
import pydicom
import pytest
from pydicom.encaps import encapsulate, generate_pixel_data_frame, get_nr_fragments
from pydicom.filebase import DicomBytesIO

import pynvjpeg as pynv


@pytest.fixture
def jpeg2k_file():
    return Path(__file__).parent / "test_jpeg2k_dicom.dcm"


def test_read():
    x = b"\xfe\xff\x00\xe0"
    exp = int("fffee000", 16)
    assert pynv._read_uint32_le(x, 0) == exp
    assert pynv._read_uint32_le(x + x, 4) == exp


@pytest.mark.parametrize(
    "bytestream,exp",
    [
        pytest.param(b"\xfe\xff\x00\xe0\x08\x00\x00\x00", True),
        pytest.param(b"\xfe\xff\x00\xe0\x00\x00\x00\x00", False),
        pytest.param(b"\x00\x00\x00\x00\x00\x00\x00\x00", False),
    ],
)
def test_has_bot(bytestream, exp):
    assert pynv._has_bot(bytestream) == exp


def test_get_num_fragments(jpeg2k_file):
    with pydicom.dcmread(jpeg2k_file) as dcm:
        actual = pynv._get_num_fragments(dcm.PixelData, len(dcm.PixelData))
        fp = DicomBytesIO(dcm.PixelData)
        fp.is_little_endian = True
        fp.seek(8)
        exp = get_nr_fragments(fp)
        assert actual == exp


def test_get_image_info(jpeg2k_file):
    with pydicom.dcmread(jpeg2k_file) as dcm:
        nf = int(dcm.NumberOfFrames)
        frame = next(iter(generate_pixel_data_frame(dcm.PixelData, nf)))
        actual = pynv.get_image_info_jpeg2k(frame, len(frame))
        assert actual["height"] == 128
        assert actual["width"] == 128
        assert actual["num_components"] == 1


def test_encode(jpeg2k_file):
    with pydicom.dcmread(jpeg2k_file) as dcm:
        num_frames, rows, cols = int(dcm.NumberOfFrames), dcm.Rows, dcm.Columns
        x = np.random.randint(0, 1024, (num_frames, rows, cols), dtype=np.uint16)
        t1 = time()
        encoded = pynv.encode_jpeg2k(x, 2)
        t2 = time()
        delta = t2 - t1
        print(f"Delta: {delta}")

        assert isinstance(encoded, list)
        assert len(encoded) == num_frames
        dcm.PixelData = encapsulate(encoded, has_bot=False)
        assert (dcm.pixel_array == x).all()


def test_decode_jpeg2k(jpeg2k_file):
    with pydicom.dcmread(jpeg2k_file) as dcm:
        nf = int(dcm.NumberOfFrames)
        actual = dcm.pixel_array
        for i, frame in enumerate(generate_pixel_data_frame(dcm.PixelData, nf)):
            decoded = pynv.decode_jpeg2k(frame, len(frame), dcm.Rows, dcm.Columns)
            assert decoded.dtype == np.uint16
            assert (decoded == actual[i]).all()


def test_decode_frames_jpeg2k(jpeg2k_file):
    with pydicom.dcmread(jpeg2k_file) as dcm:
        nf = int(dcm.NumberOfFrames)
        actual = dcm.pixel_array
        t1 = time()
        decoded = pynv.decode_frames_jpeg2k(dcm.PixelData, len(dcm.PixelData), dcm.Rows, dcm.Columns, 2)
        t2 = time()
        delta = t2 - t1
        print(f"Delta: {delta}")

        assert decoded.shape == (nf, dcm.Rows, dcm.Columns)
        assert decoded.dtype == np.uint16
        assert (decoded == actual).all()


def test_encode_decode():
    num_frames, rows, cols = 1, 512, 512
    x = np.random.randint(0, 1024, (num_frames, rows, cols), dtype=np.uint16)
    encoded = pynv.encode_jpeg2k(x, 2)
    assert isinstance(encoded, list)
    assert len(encoded) == num_frames

    decoded = pynv.decode_jpeg2k(encoded[0], len(encoded[0]), rows, cols)
    assert (decoded == x).all()
