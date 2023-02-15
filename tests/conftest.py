#!/usr/bin/env python
# -*- coding: utf-8 -*-
from pathlib import Path

import pydicom
import pytest
from pydicom.data import get_testdata_file
from pydicom.encaps import generate_pixel_data_frame


@pytest.fixture
def dicom_file_j2k_uint16():
    return Path(__file__).parent / "test_jpeg2k_dicom.dcm"


@pytest.fixture
def dicom_file_j2k_int16() -> Path:
    filename = get_testdata_file("JPEG2000.dcm")
    assert isinstance(filename, str)
    return Path(filename)


@pytest.fixture
def dicom_file_jpext() -> Path:
    filename = get_testdata_file("JPGExtended.dcm")
    assert isinstance(filename, str)
    return Path(filename)


@pytest.fixture
def dicom_file_jpl14() -> Path:
    filename = get_testdata_file("JPEG-LL.dcm")
    assert isinstance(filename, str)
    return Path(filename)


@pytest.fixture
def dcm(request, filefixture):
    filepath = request.getfixturevalue(filefixture)
    with pydicom.dcmread(filepath) as dcm:
        yield dcm


@pytest.fixture
def frame(dcm):
    nf = int(dcm.NumberOfFrames)
    frame = next(iter(generate_pixel_data_frame(dcm.PixelData, nf)))
    return frame
