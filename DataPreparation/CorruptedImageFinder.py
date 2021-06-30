# -*- coding: utf-8 -*-
"""
Created on Fri Apr  2 23:15:11 2021

"""

import os
from os import listdir
from PIL import Image

dir_path = "D:\\Authors"


for foldername in listdir(dir_path):
    for filename in listdir(os.path.join(dir_path, foldername)):
        if filename.endswith('.png'):
            try:
                img = Image.open(dir_path+"\\"+foldername+"\\"+filename) # open the image file
                img.verify() # verify that it is, in fact an image
            except (IOError, SyntaxError) as e:
                print('Bad file:', filename)
