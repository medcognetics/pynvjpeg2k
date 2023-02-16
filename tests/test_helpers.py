#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
from pydicom.encaps import generate_pixel_data_frame, get_nr_fragments
from pydicom.filebase import DicomBytesIO

import pynvjpeg as pynv


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


@pytest.mark.parametrize(
    "filefixture,exp",
    [
        ("dicom_file_j2k_uint16", 4),
        ("dicom_file_j2k_int16", 1),
    ],
)
def test_get_num_fragments(dcm, exp):
    actual = pynv._get_num_fragments(dcm.PixelData, len(dcm.PixelData))
    fp = DicomBytesIO(dcm.PixelData)
    fp.is_little_endian = True
    fp.seek(8)
    assert actual == exp == get_nr_fragments(fp)


@pytest.mark.parametrize(
    "filefixture,exp",
    [
        ("dicom_file_j2k_uint16", True),
        ("dicom_file_j2k_int16", True),
        ("dicom_file_jpext", False),
        ("dicom_file_jpl14", False),
    ],
)
def test_is_valid_jpeg2k(frame, exp):
    actual = pynv.is_valid_jpeg2k(frame, len(frame))
    assert actual == exp


@pytest.mark.parametrize(
    "filefixture",
    [
        ("dicom_file_j2k_uint16"),
        ("dicom_file_j2k_int16"),
    ],
)
def test_get_frames(dcm):
    nf = int(dcm.NumberOfFrames)
    frame_info = pynv.get_frame_offsets(dcm.PixelData, len(dcm.PixelData))
    for frame, (offset, length) in zip(generate_pixel_data_frame(dcm.PixelData, nf), frame_info):
        end = offset + length
        assert frame == dcm.PixelData[offset:end]


class TestGetImageInfo:
    @pytest.mark.parametrize(
        "filefixture,exp",
        [
            ("dicom_file_j2k_uint16", 128),
            ("dicom_file_j2k_int16", 1024),
        ],
    )
    def test_get_height(self, frame, exp):
        actual = pynv.get_image_info_jpeg2k(frame, len(frame))
        assert actual["height"] == exp

    @pytest.mark.parametrize(
        "filefixture,exp",
        [
            ("dicom_file_j2k_uint16", 128),
            ("dicom_file_j2k_int16", 256),
        ],
    )
    def test_get_width(self, frame, exp):
        actual = pynv.get_image_info_jpeg2k(frame, len(frame))
        assert actual["width"] == exp

    @pytest.mark.parametrize(
        "filefixture,exp",
        [
            ("dicom_file_j2k_uint16", 1),
            ("dicom_file_j2k_int16", 1),
        ],
    )
    def test_get_num_components(self, frame, exp):
        actual = pynv.get_image_info_jpeg2k(frame, len(frame))
        assert actual["num_components"] == exp
        assert len(actual["component_info"]) == exp

    @pytest.mark.parametrize(
        "filefixture,exp",
        [
            ("dicom_file_j2k_uint16", 0),
            ("dicom_file_j2k_int16", 1),
        ],
    )
    def test_get_component_sign(self, frame, exp):
        actual = pynv.get_image_info_jpeg2k(frame, len(frame))
        comp_info = actual["component_info"][0]
        assert comp_info["sign"] == exp

    @pytest.mark.parametrize(
        "filefixture,exp",
        [
            ("dicom_file_j2k_uint16", 16),
            ("dicom_file_j2k_int16", 16),
        ],
    )
    def test_get_component_precision(self, frame, exp):
        actual = pynv.get_image_info_jpeg2k(frame, len(frame))
        comp_info = actual["component_info"][0]
        assert comp_info["precision"] == exp
