# Learning-Monocular-Depth-by-Stereo

This is the implementation of the paper **Learning Monocular Depth by Distilling Cross-domain Stereo Networks**, ECCV 18, Xiaoyang Guo, Hongsheng Li, Shuai Yi, Jimmy Ren, and Xiaogang Wang
[\[Arxiv\]](https://arxiv.org/abs/1808.06586) [\[Demo\]](https://www.youtube.com/watch?v=QAcuYT7q_gY)

# Requirements
The code is implemented with Python 2.7 (Anaconda) (Python 3.x will cause bugs) and PyTorch 0.4.1.

    conda install pytorch=0.4.1 torchvision -c pytorch
    conda install opencv3 -c menpo
    pip install tensorboardX

# Dataset Preparation
## Scene Flow
TODO (temporarily please refer to the folder structure file and train/test list to prepare the dataset)

## KITTI Eigen Split
TODO

# Run the code
## Training
    # Step 1, pretrain the stereo model on Scene Flow datasets
    ./run_train_stereo_sceneflow.sh
    # Step 2, supervised or unsupervised fine-tune on KITTI Eigen Split
    ./run_supft100_stereo_kitti.sh
    ./run_unsupft_stereo_kitti.sh
    # Step 3, train the monocular depth model
    ./run_distill_mono.sh
## Testing
    # test the stereo model on KITTI Eigen Split, please update TEST_WORK_DIR in the bash file.
    ./run_test_stereo_kitti.sh
    # test the monocular model on KITTI Eigen Split
    ./run_test_mono_kitti.sh

The evaluation code is from [Monodepth](https://github.com/mrharicot/monodepth). We use the same crop option `--garg_crop` to evaluate the model. NOTE that we use the depth from camera view instead of LIDAR view, which is different from the default option of Monodepth.

# Pretrained Model

## Stereo Model
[StereoNoFt](https://drive.google.com/file/d/1QNLksZYkjiJ7qqxKU0a7kODVcGRebWDZ/view?usp=sharing) (Pretrained Stereo Model on Scene Flow datasets)

[StereoUnsupFt](https://drive.google.com/file/d/1qn20yitJ1zdftkX7XAms5IH-Y7QR4Z9X/view?usp=sharing)

[StereoSupFt100](https://drive.google.com/file/d/1J_fcFv2hXUtL3ADgyqw-GvTredHD8Rsc/view?usp=sharing)

## Monocular Depth Model
[StereoNoFt-->Mono](https://drive.google.com/file/d/1iiy5IfwjsOyN54u17Es29BOszRo_PI0r/view?usp=sharing)

[StereoUnsupFt-->Mono](https://drive.google.com/file/d/1xKVqusfcH4sBkVB8QzoZYBi43ApuxEU0/view?usp=sharing)

[StereoUnsupFt-->Mono pt](https://drive.google.com/file/d/1jCIle8xWHYNFY5pZ9ijac5wP5n8TWIxx/view?usp=sharing)

[StereoUnsupFt-->Mono pt C,K](https://drive.google.com/file/d/1Jxb0A3FaNZJQPFR5p8HIqpAS3pvIBZFL/view?usp=sharing)

[StereoSupFt100-->Mono](https://drive.google.com/file/d/1xmdRL4CnXtsiPlfvHfTY-XWQ1OdI_egf/view?usp=sharing)

[StereoSupFt100-->Mono pt](https://drive.google.com/file/d/1mvpVPr4_mZMrsaKWm7QSYJjbjzkJCVMm/view?usp=sharing)

[StereoSupFt100-->Mono pt C,K](https://drive.google.com/file/d/1urxvXzcSC2gyNxi5MtJxLffazeFXzCwg/view?usp=sharing)


# TODO
- [ ] Data preparation code
- [ ] check requirements
- [ ] Cityscapes pretraining code
- [ ] update use_pretrained_weights option for training monocular network. By default it is enabled.

# Citation
If you find this code useful in your research, please cite:

```
@inproceedings{guo2018learning,
  title={Learning Monocular Depth by Distilling Cross-domain Stereo Networks},
  author={Guo, Xiaoyang and Li, Hongsheng and Yi, Shuai and Ren, Jimmy and Wang, Xiaogang},
  booktitle={ECCV},
  year={2018}
}
```

# Acknowledgements
The evaluation code is from [Monodepth](https://github.com/mrharicot/monodepth). Unsupervised Monocular Depth Estimation with Left-Right Consistency, by C. Godard, O Mac Aodha, G. Brostow, CVPR 2017.

The correlation1d and resample1d modules are modified from correlation2d and resample2d modules in [flownet2-pytorch](https://github.com/NVIDIA/flownet2-pytorch).
