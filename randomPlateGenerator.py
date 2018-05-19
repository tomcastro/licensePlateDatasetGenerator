# coding: utf8

# import numpy as np
import shutil
import os
import random as r
import argparse as ap
from numberToImage import numberToImage

current_supported_letters = ['B', 'C', 'D', 'F', 'G', 'H', 'J', 'K', 'L',
                             'P', 'R', 'S', 'T', 'V', 'W', 'X', 'Y', 'Z']
old_supported_letters = ['A', 'B', 'C', 'E', 'F', 'G', 'H', 'D', 'K', 'L',
                         'N', 'O', 'P', 'R', 'S', 'T', 'U', 'V', 'X', 'Y',
                         'Z', 'W', 'M']
supported_numbers = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']


def randomPlateGenerator(style='current', reps=1):
    lps = []

    dirpath = os.getcwd()

    # Clean data directory
    shutil.rmtree(os.path.join(dirpath, 'data', 'labels'))
    shutil.rmtree(os.path.join(dirpath, 'data', 'images'))

    os.makedirs(os.path.join(dirpath, 'data', 'labels'), exist_ok=True)
    os.makedirs(os.path.join(dirpath, 'data', 'images'), exist_ok=True)

    for i in range(reps):
        licensePlate = ''

        if(style == 'current'):
            for i in range(4):
                licensePlate += r.choice(current_supported_letters)

            for i in range(2):
                licensePlate += r.choice(supported_numbers)

            # Add dot between two pairs of letters
            licensePlate = licensePlate[:2] + '  ' + licensePlate[2:]

            # Add dash between letters and numbers
            licensePlate = licensePlate[:-2] + '-' + licensePlate[-2:]

        if(style == 'old'):
            for i in range(2):
                licensePlate += r.choice(old_supported_letters)

            for i in range(4):
                licensePlate += r.choice(supported_numbers)

            # Add dot between letters and numbers
            licensePlate = licensePlate[:2] + ' ' + licensePlate[2:]

            # Add dash between two pairs of numbers
            licensePlate = licensePlate[:-2] + '-' + licensePlate[-2:]

        numberToImage(licensePlate, style)
        lps.append(licensePlate)

    return lps

if __name__ == '__main__':
    parser = ap.ArgumentParser(
            description='Generate random Chilean license plates.')
    parser.add_argument('-s', '--style', nargs=1,
                        help='license plate style (current/old)')
    parser.add_argument('-r', '--reps', nargs=1, type=int,
                        help='number of license plates to return')
    args = parser.parse_args()

    reps = args.reps[0] if args.reps else 1
    style = args.style[0] if args.style else 'current'

    randomPlateGenerator(style=style, reps=reps)
