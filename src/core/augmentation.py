from fastai.vision.all import Dihedral, Warp, Brightness, Contrast, setup_aug_tfms


def get_simple_tfms():
    aug_tfms = setup_aug_tfms([
        Dihedral(p=0.5),
        Warp(magnitude=0.2, p=0.5),
        Brightness(max_lighting=0.2, p=0.75),
        Contrast(max_lighting=0.2, p=0.75)])

    return aug_tfms
