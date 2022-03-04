## Introduction
- This is the training and evaluation code for our work "Ensemble Knowledge Guided Sub-network Search and Fine-tuning for Filter Pruning".
- Currently, the training code for ResNet for CIFAR is available.

## Paper abstract
    Conventional NAS-based pruning algorithms aim to find the sub-network with the best validation performance. However, validation performance does not successfully represent test performance, i.e., potential performance. Also, although fine-tuning the pruned network to restore the performance drop is an inevitable process, few studies have handled this issue. This paper proposes a novel sub-network search and fine-tuning method, i.e., Ensemble Knowledge Guidance (EKG). First, we experimentally prove that the fluctuation of the loss landscape is an effective metric to evaluate the potential performance. In order to search a sub-network with the smoothest loss landscape at a low cost, we propose a pseudo-supernet built by an ensemble sub-network knowledge distillation. Next, we propose a novel fine-tuning that re-uses the information of the search phase. We store the interim sub-networks, that is, the by-products of the search phase, and transfer their knowledge into the pruned network. Note that EKG is easy to be plugged-in and computationally efficient. For example, in the case of ResNet-50, about 45\% of FLOPS is removed without any performance drop in only 315 GPU hours.

<p align="center">
  <img src="https://user-images.githubusercontent.com/26036843/156765818-05517a8e-498e-4404-9445-acccbf21371d.png" width="900"><br>
  <b>Conceptual visualization of the goal of the proposed method.</b>  
</p>

## Requirement
- Tensorflow >= 2.6 (we have tested on 2.6 and 2.7)
- Pickle
- tqdm  

## How to run
1. Download Imagenet-2012 validation set at "http://www.image-net.org/challenges/LSVRC/2012/downloads".
2. Unzip the dataset to "DATA_HOME/val".
3. Download our models at "https://drive.google.com/drive/folders/1cUf8Oe_XEoGVZZ7c1SRzhfnByl47R4cW?usp=sharing".
4. Move to the codebase.
5. Train and evaluate our model by the below command.

```
  # ResNet-56 on CIFAR10
  python train_cifar.py --gpu_id 0 --arch ResNet-56 --dataset CIFAR10 --target_rate 0.45 --train_path ../test
  python test.py --gpu_id 0 --arch ResNet-56 --dataset CIFAR10 --trained_param ../test/trained_param.pkl

  # ResNet-family on ImageNet
  python test.py --gpu_id 0 --arch ResNet-{18,34,50} --dataset ILSVRC --trained_param $DOWNLOADED_PARAMS$
```

## Experimental results
<p align="center">
  <img src="https://user-images.githubusercontent.com/26036843/156766180-b67a859d-88eb-4742-b76d-be6e3810fa96.png" width="900"><br>
  <b>(Left) Potential performance vs. validation loss (right) Potential performance vs. condition number. 50 sub-networks of ResNet-56 trained on CIFAR10 were used for this experiment. accurately.</b>
</p>


<p align="center">
  <img src="https://user-images.githubusercontent.com/26036843/156767511-f8817a78-b78b-4363-9f2c-edff0fc0d82f.png" width="900"><br>
  <b>Visualization of loss landscapes of sub-networks searched by various filter importance scoring algorithms.</b>
</p>

<p align="center">
  <b>Comparison with various pruning techniques for ResNet family trained on ImageNet.</b><br>
  <img src="https://user-images.githubusercontent.com/26036843/156767848-7e9291d6-7ee3-42fa-849d-e7ebdd04273e.png" width="600">
</p>

<p align="center">
  <img src="https://user-images.githubusercontent.com/26036843/156768007-1d1a5d04-d536-4185-a81e-28a080296783.png" width="900"><br>
  <b>Performance analysis in case of ResNet-50 trained on ImageNet-2012. The left plot is the FLOPs reduction rate-Top-1 accuracy, and the right plot is the GPU hours-Top-1 accuracy.</b>
</p>

## Reference
```
@software{will_be_anounced,
  author = {Seunghyun Lee, Byun Cheol Song},
  title = {Ensemble Knowledge Guided Sub-network Search and Fine-tuning for Filter Pruning,
}
```