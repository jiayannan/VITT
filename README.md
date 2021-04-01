# VITT

## Introduction
This repo is the a codebase of the ViTT: Vision Transformer Tracker model.ViTT uses Transformer as the backbone network to build a multi-task learning model, which can detect objects and extract appearance embedding simultaneously in a single network. Our work demonstrates the effectiveness of Transformer based network in complex computer vision tasks, and paves the way for the application of pure Transformer in MOT.
## Requirements
* Python 3.6
* [Pytorch](https://pytorch.org) >= 1.2.0 
* python-opencv
* [py-motmetrics](https://github.com/cheind/py-motmetrics) (`pip install motmetrics`)
* cython-bbox (`pip install cython_bbox`)

## Test on MOT-16 Challenge
```
python track.py --cfg ./path/to/model/cfg --weights /path/to/model/weights
```

## Training instruction
- Download the training datasets.  
- Edit `cfg/ccmcpe.json`, config the training/validation combinations. A dataset is represented by an image list, please see `data/*.train` for example. 
- Run the training script:
```
python train.py --cfg ./path/to/model/cfg
```

We use 8x Nvidia Titan Xp to train the model, with a batch size of 32. You can adjust the batch size (and the learning rate together) according to how many GPUs your have. You can also train with smaller image size, which will bring faster inference time. But note the image size had better to be multiples of 32 (the down-sampling rate).

### Train with custom datasets
Adding custom datsets is quite simple, all you need to do is to organize your annotation files in the same format as in our training sets.


## Citation
If you find this repo useful in your project or research, please consider citing it:
```
@article{ViTT,
  title={ViTT: Vision Transformer Tracker},
  author={Xiaoning Zhu, Yannan Jia, Sun Jian, Zhang Pu},
  year={2019}
}
```

