## PaddlePaddle Hackathon 飞桨黑客马拉松第四期

### 149.[Highly Accurate Dichotomous Image Segmentation （ECCV 2022）](https://github.com/PaddlePaddle/Paddle/issues/ddle/Paddle/issues/51259#task149) 论文复现

### 1.目前进展

**完成了数据对齐、网络对齐、优化器对齐、loss对齐、训练对齐,精度未对齐**

### 2.训练结果

![](I:\paper_reproduction\DIS-paddleseg\log\Evaluate_f1.png)

![](I:\paper_reproduction\DIS-paddleseg\log\Evaluate_mae.png)

### 3.训练步骤

1.修改configs文件夹下的yml文件，修改数据集路径

```
2.python train.py
```

