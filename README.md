## Corresponding Paper

This project corresponds to the paper


Rui Zhang, Xuelong Li, Hongyuan Zhang, and Feiping Nie, "**Deep Fuzzy K-Means with Adaptive Loss and Entropy Regularization**," *IEEE Transactions on Fuzzy Systems*, DOI:10.1109/TFUZZ.2019.2945232.

which has been accepted in Sep, 2019. 

## Author of Code

Hongyuan Zhang and Rui Zhang

## Dependency


Now, codes of DFKM implemented by pytorch is available: 

For some reasons, the **version-1 is implemented without help of any frameworks like 

- pytorch-1.3.1
- numpy
- scikit-learn 
- scipy


## Brief Introduction

- DFKM.py: the main source code of DFKM.
- data_loader.py: load data from matlab files (*.mat). 
- utils.py: functions used in experiemnts.
- metric.py: codes for evaluation of clustering results. 

Samples to run the code is given as follows

```python
import data_loader as loader
data, labels = loader.load_data(loader.USPS)
data = data.T
for lam in [10**-3, 10**-2, 10**-1, 1]:
	print('lam={}'.format(lam))
	dfkm = DeepFuzzyKMeans(data, labels, [data.shape[0], 512, 300], lam=lam, gamma=1, batch_size=512, lr=10**-4)
	dfkm.run()
```

In fact, the data_loader.py is not necessary. You just need to input a numpy-matrix ($n\times d$) into DeepFuzzyKMeans. If you have any question, please email *hyzhang98@gmail.com* or *ruizhang8633@gmail.com*.

### Directory v0

To verify the derivations in our paper, we implement the code of DFKM only by numpy, and the related codes are put into *v0(without dl-framework)*. However, the codes are not clear enough, and they are hard to maintain and update. **So we now rewrite the core codes of DFKM.**



## Citations

```
@ARTICLE{DFKM,
  author={R. {Zhang} and X. {Li} and H. {Zhang} and F. {Nie}},
  journal={IEEE Transactions on Fuzzy Systems}, 
  title={Deep Fuzzy K-Means with Adaptive Loss and Entropy Regularization}, 
  year={2019},
  volume={},
  number={},
  pages={1-1},
}
```



## Thanks

Thanks to 
Xi Peng, Jiashi Feng, Shijie Xiao, Wei-Yun Yau,  Joey Tianyi Zhou, and Songfan Yang, "Structured AutoEncoders for Subspace Clustering", *IEEE Transactions on Image Processing*, vol. 27, no. 10, pp.5076-5086, 2018.

The codes they provide are used in our project. 
