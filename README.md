# MLCSD-Net


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

## Acknowledgement
This repository is built upon [MSDNET](https://github.com/gaohuang/MSDNet) and [RepDistiller](https://github.com/HobbitLong/RepDistiller)


