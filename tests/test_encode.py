#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
from pydicom.encaps import encapsulate

import pynvjpeg as pynv


@pytest.mark.parametrize(
    "filefixture",
    [
        ("dicom_file_j2k_uint16"),
        pytest.param("dicom_file_j2k_int16", marks=pytest.mark.xfail(reason="int16 not supported", strict=True)),
    ],
)
def test_encode(dcm):
    num_frames, _, _ = int(dcm.NumberOfFrames), dcm.Rows, dcm.Columns
    x = dcm.pixel_array
    encoded = pynv.encode_jpeg2k(x, 2)

    assert isinstance(encoded, list)
    assert len(encoded) == num_frames
    dcm.PixelData = encapsulate(encoded, has_bot=False)
    assert (dcm.pixel_array == x).all()
