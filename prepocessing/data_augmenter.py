import numpy as np
from imgaug import augmenters as iaa
from paths_and_params.params import Params
import random

params = Params()
#######################   Augmentations   ####################################
def data_augmentation(batch_imgs):
    num = random.uniform(0.3, 0.5)
    aug = iaa.Sequential([iaa.Cutout(nb_iterations=(1, 3), size=num, squared=False)])
    augs = aug.augment_images(batch_imgs)

    return augs, batch_imgs
##############################################################################