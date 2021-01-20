import os

import imgaug as ia
import numpy
from imgaug import augmenters as iaa
ia.seed(int(os.environ['PYTHONHASHSEED']))

# Defining Customized Imagedatagenerator using imgaug library
class MyAugmentor:
    def __init__(self):
        pass
    def __call__(self,input_img):
        # Sometimes(0.5, ...) applies the given augmenter in 50% of all cases,
        # e.g. Sometimes(0.5, GaussianBlur(0.3)) would blur roughly every second
        # image.
        sometimes = lambda aug: iaa.Sometimes(0.5, aug)
        input_img = numpy.array(input_img)
        seq = iaa.Sequential([
            iaa.Fliplr(0.5), # horizontal flips
            iaa.Flipud(0.2), # vertical flips

            # Small gaussian blur with random sigma between 0 and 0.5.
            # But we only blur about 50% of all images.
            sometimes(iaa.GaussianBlur(sigma=(0, 2.0))),

            # crop images by -10% to 20% of their height/width
            sometimes(iaa.CropAndPad(
                percent=(-0.1, 0.2),
                pad_mode=ia.ALL,
                pad_cval=(0, 255)
            )),

            # Apply affine transformations to some of the images
            # - scale to 80-120% of image height/width (each axis independently)
            # - translate by -20 to +20 relative to height/width (per axis)
            # - rotate by -45 to +45 degrees
            # - shear by -16 to +16 degrees
            # - order: use nearest neighbour or bilinear interpolation (fast)
            # - mode: use any available mode to fill newly created pixels
            #         see API or scikit-image for which modes are available
            # - cval: if the mode is constant, then use a random brightness
            #         for the newly created pixels (e.g. sometimes black,
            #         sometimes white)
            sometimes(iaa.Affine(
                scale={"x": (0.8, 1.5), "y": (0.8, 1.5)},
                translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
                rotate=(-45, 45),
                shear=(-16, 16),
                order=[0, 1],
                cval=(0, 255),
                mode=ia.ALL
            )),

            #drop 2-5% percent of the original size, leading to large dropped
            # rectangles.
            sometimes(iaa.CoarseDropout(
                (0.03, 0.15), size_percent=(0.02, 0.05),
                per_channel=0.2
            )),

            # Make some images brighter and some darker.
            # In 20% of all cases, we sample the multiplier once per channel,
            # which can end up changing the color of the images.
            sometimes(iaa.Multiply((0.8, 1.2), per_channel=0.2)),

            #Improve or worsen the contrast of images.
            #Comment it out after third model run (extreme saturation)
            sometimes(iaa.contrast.LinearContrast((0.75, 1.5), per_channel=0.5)),
        ],
            # do all of the above augmentations in random order
            random_order = True) # apply augmenters in random order

        output_img = seq.augment_image(input_img)
        return output_img.copy()