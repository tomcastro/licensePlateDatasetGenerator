from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw
from PIL import ImageFilter
import numpy as np
import random as r
r.seed("metricarts2018")


def numberToImage(lp, style='current'):

    # Standard size for Chilean LPs is 360x130
    canvas = Image.new('RGB', (1080, 390), 'white')

    if(style == 'current'):
        font = ImageFont.truetype('./fonts/cargo2.ttf', 200, encoding='unic')
    elif(style == 'old'):
        font = ImageFont.truetype('./fonts/helveticacond.otf',
                                  250, encoding='unic')

    draw = ImageDraw.Draw(canvas, 'RGBA')
    draw.text((100, 50), lp, 'black', font)

    #canvas = canvas.transform((1080, 390), Image.EXTENT,
    #                          (200, 50, 700, 100))

    canvas = randomlyDistortImage(canvas)

    canvas.show()


def randomlyDistortImage(img):

    width, height = img.size

    # Plane A to plane B (topL, topR, bottomR, bottomL)
    coeffs = find_coeffs(
        [(0, 0), (1080, 0), (1080, 390), (0, 390)],
        [(r.randint(0, 100), r.randint(0, 50)),
         (r.randint(1000, 1080), r.randint(0, 50)),
         (r.randint(900, 1080), r.randint(280, 390)),
         (r.randint(0, 100), r.randint(250, 390))])

    # Perspective transform
    img = img.transform((width + 180, height + 65),
                        Image.PERSPECTIVE, coeffs, Image.BICUBIC)

    img = img.filter(ImageFilter.GaussianBlur(r.randint(0, 100)))


    return img


def find_coeffs(pa, pb):
    matrix = []
    for p1, p2 in zip(pa, pb):
        matrix.append([p1[0], p1[1], 1, 0, 0, 0, -p2[0]*p1[0], -p2[0]*p1[1]])
        matrix.append([0, 0, 0, p1[0], p1[1], 1, -p2[1]*p1[0], -p2[1]*p1[1]])

    A = np.matrix(matrix, dtype=np.float)
    B = np.array(pb).reshape(8)

    res = np.dot(np.linalg.inv(A.T * A) * A.T, B)
    return np.array(res).reshape(8)
