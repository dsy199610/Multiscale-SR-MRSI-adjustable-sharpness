# Multi-scale Super-resolution Magnetic Resonance Spectroscopic Imaging with Adjustable Sharpness (MICCAI 2022)

Siyuan Dong, Gilbert Hangel, Wolfgang Bogner, Georg Widhalm, Karl Rössler, Siegfried Trattnig, Chenyu You, Robin de Graaf, John A Onofrey, James S Duncan

[[Paper Link](https://link.springer.com/chapter/10.1007/978-3-031-16446-0_39)]

### Citation
If you use this code please cite:

    @inproceedings{dong2022multi,
      title={Multi-scale super-resolution magnetic resonance spectroscopic imaging with adjustable sharpness},
      author={Dong, Siyuan and Hangel, Gilbert and Bogner, Wolfgang and Widhalm, Georg and R{\"o}ssler, Karl and Trattnig, Siegfried and You, Chenyu and de Graaf, Robin and Onofrey, John A and Duncan, James S},
      booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
      pages={410--420},
      year={2022},
      organization={Springer}
    }
   
### Environment and Dependencies
 Requirements:
 * python 3.7.11
 * pytorch 1.1.0
 * pytorch-msssim 0.2.1
 * torchvision 0.3.0
 * numpy 1.19.2

### Directory
    main.py                             # main file for training and evaluation
    loader
    └──  dataloader.py                  # dataloader
    utils
    ├──logs.py                          # logging
    └──utils.py                         # utility files
    models
    ├──MUNet.py                         # single-scale network
    ├──MUNet_AMLayer.py                 # AMLayer
    ├──MUNet_HyperNetworks.py           # HyperNetworks
    ├──MUNet_FilterScaling.py           # Filter Scaling
    ├──MUNet_FilterScaling_Met.py       # Filter Scaling with Met
    ├──MUNet_FilterScaling_Met_adv.py   # Filter Scaling with Met + adjustable sharpness
    └──cWGAN.py                         # functions for training cWGAN
