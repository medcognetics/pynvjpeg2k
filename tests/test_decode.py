#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pytest
from pydicom.encaps import generate_pixel_data_frame

import pynvjpeg as pynv


@pytest.mark.parametrize(
    "filefixture",
    [
        ("dicom_file_j2k_uint16"),
        pytest.param("dicom_file_j2k_int16", marks=pytest.mark.xfail(reason="int16 not supported", strict=True)),
        pytest.param("dicom_file_jpext", marks=pytest.mark.xfail(reason="not JPEG2000", strict=True)),
    ],
)
def test_decode_jpeg2k(dcm):
    nf = int(dcm.NumberOfFrames)
    actual = dcm.pixel_array
    for i, frame in enumerate(generate_pixel_data_frame(dcm.PixelData, nf)):
        decoded = pynv.decode_jpeg2k(frame, len(frame), dcm.Rows, dcm.Columns)
        assert (decoded == actual[i]).all()


@pytest.mark.parametrize(
    "filefixture",
    [
        ("dicom_file_j2k_uint16"),
        pytest.param("dicom_file_j2k_int16", marks=pytest.mark.xfail(reason="int16 not supported", strict=True)),
        pytest.param("dicom_file_jpext", marks=pytest.mark.xfail(reason="not JPEG2000", strict=True)),
    ],
)
def test_decode_frames_jpeg2k(dcm):
    nf = int(dcm.NumberOfFrames)
    actual = dcm.pixel_array
    decoded = pynv.decode_frames_jpeg2k(dcm.PixelData, len(dcm.PixelData), dcm.Rows, dcm.Columns, 2)
    assert decoded.shape == (nf, dcm.Rows, dcm.Columns)
    assert (decoded == actual).all()


def test_encode_decode():
    num_frames, rows, cols = 1, 512, 512
    x = np.random.randint(0, 1024, (num_frames, rows, cols), dtype=np.uint16)
    encoded = pynv.encode_jpeg2k(x, 2)
    assert isinstance(encoded, list)
    assert len(encoded) == num_frames

    decoded = pynv.decode_jpeg2k(encoded[0], len(encoded[0]), rows, cols)
    assert (decoded == x).all()
