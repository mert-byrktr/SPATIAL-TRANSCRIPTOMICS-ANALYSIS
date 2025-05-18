# SPATIAL-TRANSCRIPTOMICS-ANALYSIS

Input: H&E images with spot coordinates.

For each spot (~55 μm in diameter), the goal is to predict the abundances (deconvolution estimates) of 35 cell types using image data alone.

For each spot, 224x224 patches were extracted centering these spots and used as inputs to the models.

Transformations used:

```python
train_transforms = A.Compose([

    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),

    A.Affine(
        scale=(0.9, 1.1),
        translate_percent=(-0.1, 0.1),
        rotate=(-25, 25),
        shear=(-7, 7),
        p=0.9
    ),
    A.OneOf([
        A.MotionBlur(blur_limit=5),
        A.MedianBlur(blur_limit=5),
        A.GaussianBlur(blur_limit=5),
        A.GaussNoise(var_limit=(5.0, 30.0)),
      ], p=0.7),
    A.OneOf([
        A.OpticalDistortion(distort_limit=1.0),
        A.GridDistortion(num_steps=5, distort_limit=1.),
        A.ElasticTransform(alpha=3),
      ], p=0.7),


    A.CLAHE(clip_limit=4.0, p=0.7),
    A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.5),
    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, p=0.5),


    A.Normalize(mean=norm_mean, std=norm_std),
    ToTensorV2()
])

valid_transforms = A.Compose([
    # A.Resize(CONFIG.img_size, CONFIG.img_size),
    A.Normalize(mean=norm_mean, std=norm_std),
    ToTensorV2()
])
```

Also random CutMix applied to stabilize the training and increase the generalizability.

**Loss function:** Target cell type abundances for each spot were first rank-transformed and scaled to [0,1]. This approach helps normalize the target distributions across different spots/slides, makes the model less sensitive to extreme outlier values in abundances, and focuses the learning on the relative ordering of cell type prevalence. L1Loss was then applied to these transformed targets, with a Sigmoid activation on the model's output layer.

**Metric:** Spearman's Rank Correlation

**Validation:** Each slide used as validation. (Leave-one-out validation.)

**Models used**: Convnext-tiny, efficientnet_v2_m, resnext, regnet, swin transformers

**Optimizer and scheduler**: AdamW optimizer and CosineAnnealingLR Scheduler.

**Methods that did not work:**
* Using differentiable version of spearmans rank corr. as loss function
* Using pearson correlation as loss function
* Mixup Transform
* Stain Normalization
* Graph Neural Networks
* Sqrt - log1p transformation on targets
* Treating target as probability distribution and using KL-Divergence as loss.
* Normalizing targets
* Stochastic Weight Averaging and EMA
