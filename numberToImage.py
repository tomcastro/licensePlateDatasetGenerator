from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw
from PIL import ImageFilter
from imutils import contours, object_detection

import matplotlib.pyplot as plt
import numpy as np
import random as r
import cv2
import os
import copy

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

    # canvas = canvas.transform((1080, 390), Image.EXTENT,
    #                          (200, 50, 700, 100))

    canvas = randomlyDistortImage(canvas)
    getBoundingBoxes(canvas, lp)

    # canvas.show()


def getBoundingBoxes(img, lp):

    cv2_img = np.array(img)
    cv2_original_image = copy.deepcopy(cv2_img)
    cv2_img_grey = cv2.cvtColor(cv2_img, cv2.COLOR_RGB2GRAY)

    ret, thresh = cv2.threshold(cv2_img_grey, 127, 255, 0)
    im2, cnts, h = cv2.findContours(thresh, 1, 2)

    cnts, boundingBoxes = contours.sort_contours(cnts, method='left-to-right')

    bb = np.asarray(boundingBoxes)

    # Eliminate greater (first) bounding box
    bb = bb[1:]

    # Convert to TL, TR, BR, BL style and back, and eliminate
    # smaller boxes for letters that contain more than one
    bb[:, 2:] = bb[:, 2:] + bb[:, :2]
    bb = object_detection.non_max_suppression(bb)
    bb[:, 2:] = bb[:, 2:] - bb[:, :2]

    # Sort bounding boxes by X position (left-to-right)
    bb = bb[np.argsort(bb[:, 0])]

    bb_text = str(len(bb)) + '\n'
    lp = ''.join(lp.split())

    for i, box in enumerate(bb):
        rectangle(cv2_img, (255, 0, 0), 3, *box)
        bb_text += str(box[0]) + ' ' + str(box[1]) + ' ' + str(box[2]) + ' ' \
            + str(box[3]) + ' ' + lp[i] + '\n'

    print(bb_text)

    # Save label files
    i = 0
    while os.path.exists('data/labels/license_plate_img_%s.txt' % i):
        i += 1

    file = open('data/labels/license_plate_img_%s.txt' % i, 'w+')
    file.write(bb_text)
    file.close()

    cv2.imwrite('data/images/license_plate_img_%s.jpg' % i, cv2_original_image)

    # plt.subplot(111)
    # plt.imshow(cv2_img)

    # plt.show()


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


def rectangle(image, color, thickness, x, y, w, h, label=None):
    """Draw a rectangle.
    Parameters
    ----------
    x : float | int
        Top left corner of the rectangle (x-axis).
    y : float | int
        Top let corner of the rectangle (y-axis).
    w : float | int
        Width of the rectangle.
    h : float | int
        Height of the rectangle.
    label : Optional[str]
        A text label that is placed at the top left corner of the
        rectangle.
    """
    pt1 = int(x), int(y)
    pt2 = int(x + w), int(y + h)
    cv2.rectangle(image, pt1, pt2, color, thickness)
    if label is not None:
        text_size = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_PLAIN, 1, thickness)

        center = pt1[0] + 5, pt1[1] + 5 + text_size[0][1]
        pt2 = pt1[0] + 10 + text_size[0][0], pt1[1] + 10 + \
            text_size[0][1]
        cv2.rectangle(image, pt1, pt2, color, -1)
        cv2.putText(image, label, center, cv2.FONT_HERSHEY_PLAIN,
                    1, (255, 255, 255), thickness)
