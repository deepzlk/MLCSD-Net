




#=0 python train_backbone.py --model ResNet18 --dataset cifar100 -r 1  --trial 0
#CUDA_VISIBLE_DEVICES=0 python train_menm.py --model ResNet18 --dataset cifar100 -r 1  --trial 0

CUDA_VISIBLE_DEVICES=0 python train_self.py --model ResNet18 --dataset cifar100 -r 1  -a 1 --trial 0
CUDA_VISIBLE_DEVICES=0 python train_self.py --model ResNet18 --dataset cifar100 -r 1  -a 5 --trial 1

#CUDA_VISIBLE_DEVICES=0 python train_mlcsd.py --model ResNet18 --dataset cifar100 -r 1  -a 5 -b 10 --trial 0