# MLCSD-Net


Official PyTorch implementation for the following paper:

Our implementation is based upon [MSDNET](https://github.com/gaohuang/MSDNet) and [RepDistiller](https://github.com/HobbitLong/RepDistiller)
---
<p align="center">
<img src="https://user-images.githubusercontent.com/29576696/161406403-15c6da87-cd09-4203-adaa-09cc2badf1c1.jpeg" width=100% height=100% 
class="center">
</p>




## Train

**Multi-Exit Network (MEN)**

````bash
python train_backbone.py \
````


**Multi-Exit Network (MEN)**

````bash
python train_menm.py \
````


**Self-Distillation Network (SDN)**

````bash
python train_self.py \
````


**Multi-Level Collaborative Self-Distillation Network (MLCSD-Net)**

````bash
python train_mlcsd.py \
````

Evaulation results will be generated at the end of training.

- `result.txt`: contains mIOU for each exit and the average mIOU of the four exits. 

- `test_stats.json`: contains FLOPs and number of parameters.

- `final_state.pth`: the trained model file.

- `config.yaml`: the configuration file. 

## Test

**Evaluation** 

```
python tools/test_ee.py --cfg <Your output directoy>/config.yaml
```

## Acknowledgement
This repository is built upon [HRNet-Semantic-Segmentation](https://github.com/HRNet/HRNet-Semantic-Segmentation/tree/pytorch-v1.1).

## License
This project is released under the MIT license. Please see the [LICENSE](LICENSE) file for more information.

## Citation
If you find this repository helpful, please consider citing:
```
@Article{liu2022anytime,
  author  = {Zhuang Liu and Zhiqiu Xu and Hung-Ju Wang and Trevor Darrell and Evan Shelhamer},
  title   = {Anytime Dense Prediction with Confidence Adaptivity},
  journal = {International Conference on Learning Representations (ICLR)},
  year    = {2022},
}
```
