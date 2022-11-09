# PyNVJPEG2K

PyNVJPEG2K is a work in progress Python library to accelerate decompression and compression of JPEG2000 encoded images.
Python bindings have not yet been implemented, but there is support for accelerated decoding using a standaclone C++ program.
Only decompression of single channel inputs has been implemented.

This library depends on `libnvjpeg2k`, which is provided only to developers in the Nvidia developer program. A copy
of `libnvjpeg2k` has been provided with this repository since PyNVJPEG2K is currently a private MedCognetics repository.


## Usage

### Encoding

Only 16 bit grayscale inputs are supported. Only lossless encoding is supported. Number of frames must be divisible by the batch size.

Encoding a 8x512x512 3D volume w/ batch size 8
```python
import pynvjpeg as pynv

num_frames, rows, cols = 8, 512, 512 
batch_size = 4
data = np.random.randint(0, 1024, (num_frames, rows, cols), dtype=np.uint16)
encoded = pynv.encode_jpeg2k(x, batch_size)
assert len(encoded) == num_frames
assert all(isinstance(f, bytes) for f in encoded)
```

### Decoding

Only 16 bit grayscale images are supported. Number of frames must be divisible by the batch size. The decoding process assumes that the output
image dimensions (rows, columns, number of frames) are known ahead of time. For DICOMs, this information will be present in the metadata.

Single frame decoding
```python
import pynvjpeg as pynv
decoded = pynv.decode_jpeg2k(frame, len(frame), rows, cols)
```

Multi-frame batched decoding
```python
import pynvjpeg as pynv

# number of decoded frames must be divisible by batch size.
# it is assumed that the number of frames is known ahead of time
batch_size = 4
decoded = pynv.decode_frames_jpeg2k(data, len(data), rows, columns, 4)
```

# TODO 
* Can we relax the constraint that image dimensions are known ahead of time without much performance hit?
* Can we relax the constraint that batch size be a divsior of the number of frames?
* Can we accelerate JPEG Lossless transfer syntaxes?
