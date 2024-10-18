# Awesome-CLIP


## Improvement & Innovation





## Train & Finetune

| Title | Abstract | Intro | Useful Links |
|:----| :---:| :----: | :---:|
|2021|
| [![Star](https://img.shields.io/github/stars/gaopengcuhk/CLIP-Adapter.svg?style=social&label=Star)](https://github.com/gaopengcuhk/CLIP-Adapter) <br> **CLIP-Adapter: Better Vision-Language Models with Feature Adapters** <br>| CLIP Adapter是一个为few-shot classfication任务设计的一个插入式模块.在冻住的clip特征上添加一个残差连接的微调器，使得CLIP能更好地应对分类等下游任务.|<img src="./images/CLIP-Adapter.jpg"  width="1280px"/>| [[Github](https://github.com/gaopengcuhk/CLIP-Adapter)] <br> [[Paper](https://arxiv.org/pdf/2110.04544)] |



## Zero-Shot

| Title | Abstract | Intro | Useful Links |
|:----| :---:| :----: | :---:|
|2022|
| [![Star](https://img.shields.io/github/stars/KaiyangZhou/CoOp.svg?style=social&label=Star)](https://github.com/KaiyangZhou/CoOp) <br> **CALIP: Zero-Shot Enhancement of CLIP with Parameter-free Attention** <br>| 文章出发点是如何在不finetune的情况下，提升clip在下游任务上的zero-shot能力.文章提出了一种parameter-free的注意力模块(CALIP)，引导视觉和文本表示相互交互，并通过注意力探索跨模式信息特征.通过这种方式，图像与文本两个模态特征相互感知，以实现更好的自适应零样本对齐.|<img src="./images/CoOp.png"  width="1280px"/>| [[Github](https://github.com/KaiyangZhou/CoOp)] <br> [[Paper](https://arxiv.org/pdf/2109.01134)] |
| [![Star](https://img.shields.io/github/stars/ZiyuGuo99/CALIP.svg?style=social&label=Star)](https://github.com/ZiyuGuo99/CALIP) <br> **CoOp: Learning to Prompt for Vision-Language Models** <br>| 受NLP领域prompt learning的启发，文章提出了Context Optimization(CoOp)，用于将类CLIP式的视觉语言模型迁移到下游图像识别任务.具体而言，CoOp将预训练模型参数freeze，使用可学习向量对提示的上下文单词进行建模.作者在11个下游任务上验证CoOp的有效性，结果显示CoOp的性能明显好于原始预训练模型如CLIP.|<img src="./images/CALIP.png"  width="1280px"/>| [[Github](https://github.com/ZiyuGuo99/CALIP)] <br> [[Paper](https://arxiv.org/pdf/2209.14169)] |


## Retrieval

## Captioning

| Title | Abstract | Intro | Useful Links |
|:----| :---:| :----: | :---:|
|2021|
| [![Star](https://img.shields.io/github/stars/rmokady/CLIP_prefix_caption.svg?style=social&label=Star)](https://github.com/rmokady/CLIP_prefix_caption) <br> **ClipCap: CLIP Prefix for Image Captioning** <br>| 作者提出了CLIPCap模型来生成image captions.具体而言，借助CLIP提取图像embeadding，训练一个mapping network，为每一个caption生成前缀.直接和caption embedding 做结合（concatenation），形成新的embedding，送入GPT-2生成captions. |<img src="./images/ClipCap.png"  width="1280px"/>| [[Github](https://github.com/rmokady/CLIP_prefix_caption)] <br> [[Paper](https://arxiv.org/pdf/2111.09734)] |

##


