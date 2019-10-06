## Corresponding Paper

This project correspond to the paper

Rui Zhang, Xuelong Li, Hongyuan Zhang, and Feiping Nie, "**Deep Fuzzy K-Means with Adaptive Loss and Entropy Regularization**," *IEEE Transactions on Fuzzy System* 

which has been accept in Sep, 2019. 

## Author of Code

Hongyuan Zhang and Rui Zhang

## Dependence

For some reasons, the **version-1 is implemented without help of any frameworks like caffe, tensorflow and so on**. The only third-party packages you need is several well-known ones, including: 

- numpy
- pickle
- scikit-learn 
- scipy

**To achieve more convinience and efficiency, we will reimplement DFKM by certain deep learning framework(tensorflow/caffe/pytorch). Due to the schedule of research, the code will be implemented before Mar, 2020.**

## Brief Introduction

- fuzzy_k_means.py: the main source code of DFKM.
- image_seg.py: code to perform image segmentation. 
- utils.py: functions used in experiemnts.
- kernel_k_means.py/RobustFKM.py/fuzzy_k_means.py: codes of competitors which are implemented by ourselves. 
- usages: MATLAB codes that used in our experiments.

You can test the code by the following command

```shell
python jaffe_test.py
```

and you can imitate it to run DFKM on other datasets. 



## Thanks

Thanks to Xi Peng, Jiashi Feng, Shijie Xiao, Wei-Yun Yau,  Joey Tianyi Zhou, and Songfan Yang, "Structured AutoEncoders for Subspace Clustering", *IEEE Transactions on Image Processing*, vol. 27, no. 10, pp.5076-5086, 2018.

The codes they provided are used in our project. 
