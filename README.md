## Introduction

This  reposity contains the train code for our work **Diverse Instance Discovery: Vision-Transformer for Instance-Aware Multi-Label Image Recognition**. Our goal is to leverage ViT's patch tokens and self-attention mechanism to mine rich instances in multi-label images, named diverse instance discovery (DiD). To this end, we propose a semantic category-aware module and a spatial relationship-aware module, respectively, and then combine the two by a re-constraint strategy to obtain instance-aware attention maps. Finally, we propose a weakly supervised object localization-based approach to extract multi-scale local features, to form a multi-view pipeline. Our method requires only weakly supervised information at the label level, no additional knowledge injection or other strongly supervised information is required. Experiments on three benchmark datasets show that our method significantly outperforms previous works and achieves state-of-the-art results under fair experimental comparisons.

### My Paper Title

**Diverse Instance Discovery: Vision-Transformer for Instance-Aware Multi-Label Image Recognition**

## Results
#### Results on MS-COCO

![image-20210910225321978](./assets/image-20210910225321978.png)

#### Results on PASCAL VOC 2007 and PASCAL VOC2012

![image-20210910225457731](./assets/image-20210910225457731.png)

## Usage

#### Train on MS-COCO

To run this code, you need eight Tesla V100 GPU (32G memory $\times$ 8). Make sure a floder named 'augments_DP' has been created.
```
sh DP_run.sh
```
## Related work

[TS-CAM](https://arxiv.org/abs/2103.14862)

[MCAR](https://arxiv.org/abs/2103.14862)

