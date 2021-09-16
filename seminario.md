---
marp: true
---

<H1> Semantic Segmentation Pytorch </H1>

## Schema
1. CRISP-DM
2. Data Understanding
3. Data Preparation
4. Modeling
5. Evaluation
6. Deployment

## 4. Modeling - Stochastic Gradient Descent

Optimization problem in Machine Learning boils down to:

$D = \{ (x_{i}, y_{i}) \}_{i = 1}^{n} \in \R^{d} \times Y$

where *d* = dimension and *n* = observations.

Least-Squares:

$\frac{1}{n}\| Ax - b \|_{2}^{2} = \frac{1}{n}\sum_{i = 1}^{n}(a_{i}^{T}x - b_{i})^{2}$

$\frac{1}{n}\sum_{i = 1}^{n}(a_{i}^{T}x - b_{i})^{2} + \lambda\sum_{j = 1}^{d}|x_{j}|$

$\frac{1}{n}\sum_{i = 1}^{n}(a_{i}^{T}x - b_{i})^{2} + \lambda\sum_{j = 1}^{d}|x_{j}|^{2}$



Support Vector Machine

$\frac{1}{2} \| x \|^{2} + \frac{c}{n}\sum_{i = 1}^{n} max[ 0, 1 - y_{i}(x^{T}a_{i} + b)]$

Deep Neural Networks
$\frac{1}{n}\sum_{i = 1}^{n}L(y_{i}, DNN(x, a_{i}))$

### Basic Gradient Descent

$x_{k+1} = x_{k} - \nabla f(x) = x_{k} - \alpha_{k} \frac{1}{n} \sum_{i = 1}^{n} \nabla f_{i} (x)$
 
*n* is too big, hence computationally to compute gradient.

Solution: Randomly pick a set of integers $i(k) \in \{ 1,2,3,....,n\}$ to sample the gradient (mini-batch).

$x_{k+1} = x_{k} - \alpha_{k} \frac{1}{n} \sum_{i = 1}^{n} \nabla f_{i(k)}(x_{k})$

Stochastic Gradient is an unbiased estimator of the true gradient.

## 5. Pytorch

Pytorch has two primitives to work with data: **torch.utils.data.DataLoader**and **torch.utils.data.Dataset**. **Dataset** stores the samples and their corresponding labels, and **DataLoader** wraps an iterable around the **Dataset**.

Pytorch offers domain specific libraries such as **TorchText**, **TorchVision**, and **TorchAudio**, all of which include datasets and pre-trained models.

    from torchvision import models
    dir(models)

>['AlexNet',
> 'DenseNet',
> 'GoogLeNet',
> 'GoogLeNetOutputs',
> 'Inception3',
> 'InceptionOutputs',
> 'MNASNet',
> 'MobileNetV2',
> 'MobileNetV3',
> 'ResNet',
> 'ShuffleNetV2',
> 'SqueezeNet',
> 'VGG',
> '_GoogLeNetOutputs',
> '_InceptionOutputs'...]

    alexnet = models.AlexNet()
    resnet = models.resnet101(pretrained=True)


## Torch.Hub

Pytorch Hub is a pre-trained model repository designed to facilitate research reproducibility. 
Pytorch Hub supports publishing pre-trained models(model definitions and pre-trained weights) to a github repository by adding a simple hubconf.py file.

https://pytorch.org/docs/stable/hub.html#publishing-models


    import torch 
    from torch import hub
    resnet18_model = hub.load('pytorch/vision:master', 
                              'resnet18',      
                              pretrained=True)

## Tensors

Tensors are similar tu NumPy's ndarrays, except that tensors can run on GPUs or other hadware accelerators (TPU). 

Tensors and NumPy arrays can often share the same underlying memory, eliminating the need to copy data.

Tensors are also optimized for automatic differentiation 

Tensors can ber converted to Numpy arrays and vicebersa.

    import torch
    import numpy as np 

    a_tensor = torch.ones(3)
    a_numpy = a_tensor.numpy()
    a_numpy[2] = 7.
    b_tensor = torch.from_numpy(a_numpy)

    print(a_tensor)
    print(a_numpy)
    print(b_tensor)

>tensor([1., 1., 7.])
>
>[1. 1. 7.]
>
>tensor([1., 1., 7.])

Attributes of a tensor:

    print(f"Shape of tensor: {b_tensor.shape}")
    print(f"Datatype of tensor: {b_tensor.dtype}")
    print(f"Device tensor is stored on: {b_tensor.device}")

>Shape of tensor: torch.Size([3])
>
>Datatype of tensor: torch.float32
>
>Device tensor is stored on: cpu

    img_t = torch.randn(3,5,5)      #shape = (channels, rows, columns)
    weights = torch.tensor([0.2126, 0.7152, 0.0722])
    batch_t = torch.randn(2, 3, 5, 5)   #shape (batch, channels, rows, columns)
    img_gray_naive = img_t.mean(-3)

    batch_gray_naive = batch_t.mean(-3)
    img_gray_naive.shape, batch_gray_naive.shape

>(torch.Size([5, 5]), torch.Size([2, 5, 5]))

