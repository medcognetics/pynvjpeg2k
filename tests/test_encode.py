#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pydicom
import pytest
from pydicom.encaps import encapsulate

import pynvjpeg as pynv


@pytest.mark.parametrize(
    "filefixture",
    [
        ("dicom_file_j2k_uint16"),
        ("dicom_file_j2k_int16"),
        ("dicom_file_j2k_2d"),
    ],
)
def test_encode(request, filefixture):
    filepath = request.getfixturevalue(filefixture)
    dcm = pydicom.dcmread(filepath)
    num_frames, _, _ = int(dcm.get("NumberOfFrames", 1)), dcm.Rows, dcm.Columns
    x = dcm.pixel_array
    H, W = x.shape[-2:]
    encoded = pynv.encode_jpeg2k(x.reshape(num_frames, H, W), 2)

    assert isinstance(encoded, list)
    assert len(encoded) == num_frames
    dcm = pydicom.dcmread(filepath)
    dcm.PixelData = encapsulate(encoded, has_bot=False)
    assert (dcm.pixel_array == x).all()
