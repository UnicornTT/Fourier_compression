import numpy as np
from scipy.fftpack import dct
from scipy.fftpack import idct
from PIL import Image


def dct_encode(matrix):
    intermediate = dct(matrix, type=2, norm="ortho", axis=0)
    final = dct(intermediate, type=2, norm="ortho", axis=1)
    return final


def dct_decode(matrix):
    intermediate = idct(matrix, type=2, norm="ortho", axis=0)
    final = idct(intermediate, type=2, norm="ortho", axis=1)
    return final


def compress_dct(matrix, new_size):
    return matrix[0:new_size, 0:new_size]


def pad_dct(matrix, new_size):
    temp = np.zeros((new_size, new_size))
    rows, cols = matrix.shape
    temp[0:rows, 0:cols] = matrix
    return temp


def decode_image(dct_matrix, image_size):
    raw_image_matrix = dct_decode(pad_dct(dct_matrix, image_size))
    image = Image.fromarray(np.clip(np.rint(raw_image_matrix), 0, 255).astype('uint8'), mode="L")
    return image


def encode_image(image, compressed_size):
    return compress_dct(dct_encode(np.asarray(image)), compressed_size)


i = Image.open('test_image.bmp').convert("L")
i.save('gray.jpg')
dct_matrix = encode_image(i, 200)
j = decode_image(dct_matrix, 400)
j.save('compressed.jpg')
