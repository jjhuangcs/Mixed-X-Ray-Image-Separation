# [Mixed X-Ray Image Separation for Artworks with Concealed Designs](https://ieeexplore.ieee.org/abstract/document/9810038)

<!--![visitors](https://visitor-badge.glitch.me/badge?page_id=jjhuangcs/WINNet)-->

Pu Wei#, [Jun-Jie Huang](https://jjhuangcs.github.io/)#* (jjhuang@nudt.edu.cn), Barak Sober, Nathan Daly, Catherine Higgitt, Ingrid Daubechies, [Pier Luigi Dragotti](http://www.commsp.ee.ic.ac.uk/~pld/) and 
Miguel Rodrigues (#co-first author, *corresponding author)

The proposed separation network consists of two components: the analysis and the synthesis sub-networks. The analysis sub-network is based on learned coupled iterative shrinkage thresholding algorithms (LCISTA) designed using algorithm unrolling techniques, and the synthesis sub-network consists of several linear mappings. The learning algorithm operates in a totally self-supervised fashion without requiring a sample set that contains both the mixed X-ray images and the separated ones.

Pytorch implementation for "Mixed X-Ray Image Separation for Artworks with Concealed Designs" (TIP'2022).


# 1. Dependencies
* Python
* torchvision
* PyTorch>=1.0
* OpenCV for Python
* HDF5 for Python
* tensorboardX (Tensorboard for Python)

# 2. Usage

```python XraySepNet_test.py --visible filename_visible_image --xray filename_xray_image --gray filename_grayscale_image```

# Citation

If you use any part of this code in your research, please cite our paper:

```
@article{huang2022MixSep,
  author={Pu, Wei and Huang, Jun-Jie and Sober, Barak and Daly, Nathan and Higgitt, Catherine and Daubechies, Ingrid and Dragotti, Pier Luigi and Rodrigues, Miguel R. D.},
  journal={IEEE Transactions on Image Processing}, 
  title={Mixed X-Ray Image Separation for Artworks With Concealed Designs}, 
  year={2022},
  volume={31},
  number={},
  pages={4458-4473},
  doi={10.1109/TIP.2022.3185488}}
```


