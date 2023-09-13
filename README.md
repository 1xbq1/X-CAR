# X-CAR
X-Invariant Contrastive Augmentation and Representation Learning for Semi-Supervised Skeleton-Based Action Recognition
## Requirements
- python == 3.8.3
- pytorch == 1.11.0
- CUDA == 11.2
## Data Preparation
Download the raw data of [NTU-RGB+D](https://github.com/shahroudy/NTURGB-D), and [NW-UCLA](https://www.dropbox.com/s/10pcm4pksjy6mkq/all_sqe.zip?dl=0).
## Training
```
python main.py --config 1001 --generate_label --no_progress_bar
python main.py -c 1013 
```
## Acknowledgements
This repo is based on [PA-ResGCN](https://gitee.com/yfsong0709/ResGCNv1), thanks to the original authors for their works!
## Citation
Please cite the following paper if you use this repository in your reseach.
```
@article{xu2022x,
  title={X-invariant contrastive augmentation and representation learning for semi-supervised skeleton-based action recognition},
  author={Xu, Binqian and Shu, Xiangbo and Song, Yan},
  journal={IEEE Transactions on Image Processing},
  volume={31},
  pages={3852--3867},
  year={2022}
}
 ```
