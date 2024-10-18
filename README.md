# Awesome-CLIP


## Improvement & Innovation

| Title | Abstract | Intro | Useful Links |
|:----| :---:| :----: | :---:|
|2022|
| [![Star](https://img.shields.io/github/stars/FlagAI-Open/FlagAI.svg?style=social&label=Star)](https://github.com/FlagAI-Open/FlagAI) <br> **AltCLIP: Altering the Language Encoder in CLIP for Extended Language Capabilities** <br>|文章提出一个概念上简单而有效的方法来训练一个强大的双语/多语多模态表示模型. 使用预训练的多语言文本编码器XLMR替换Clip的文本编码器，通过两阶段的训练模式(Teacher Learning; Contrastive Learning)对齐文本和图像表征. |<img src="./images/AltCLIP.png”  width="1280px"/>| [[Github](https://github.com/FlagAI-Open/FlagAI)] <br> [[Paper](https://arxiv.org/pdf/2211.06679)] |



## Train & Finetune

| Title | Abstract | Intro | Useful Links |
|:----| :---:| :----: | :---:|



## Zero-Shot & Few-Shot

| Title | Abstract | Intro | Useful Links |
|:----| :---:| :----: | :---:|
|2021|
| [![Star](https://img.shields.io/github/stars/gaopengcuhk/CLIP-Adapter.svg?style=social&label=Star)](https://github.com/gaopengcuhk/CLIP-Adapter) <br> **CLIP-Adapter: Better Vision-Language Models with Feature Adapters** <br>| CLIP Adapter是一个为few-shot classfication任务设计的一个插入式模块.在冻住的clip特征上添加一个残差连接的微调器，使得CLIP能更好地应对分类等下游任务.|<img src="./images/CLIP-Adapter.jpg"  width="1280px"/>| [[Github](https://github.com/gaopengcuhk/CLIP-Adapter)] <br> [[Paper](https://arxiv.org/pdf/2110.04544)] |
|2022|
| [![Star](https://img.shields.io/github/stars/gaopengcuhk/Tip-Adapter.svg?style=social&label=Star)](https://github.com/gaopengcuhk/Tip-Adapter) <br> **Tip-Adapter: Training-free Adaption of CLIP** <br>|为了提高Clip的few-shot能力，文章提出一种免训练的方法，、名为Tip-Adapter. 通过从少样本监督中构建query-key缓存模型来获取适配器的权重. 通过缓存模型，与传统的finetune方法相比，Tip-Adapter表现出极高的效率. |<img src="./images/Tip-Adapter.png"  width="1280px"/>| [[Github](https://github.com/gaopengcuhk/Tip-Adapter)] <br> [[Paper]( https://arxiv.org/pdf/2207.09519)]  |
| [![Star](https://img.shields.io/github/stars/ZiyuGuo99/CALIP.svg?style=social&label=Star)](https://github.com/ZiyuGuo99/CALIP) <br> **CALIP: Zero-Shot Enhancement of CLIP with Parameter-free Attention** <br>| 文章出发点是如何在不finetune的情况下，提升clip在下游任务上的zero-shot能力.文章提出了一种parameter-free的注意力模块(CALIP)，引导视觉和文本表示相互交互，并通过注意力探索跨模式信息特征.通过这种方式，图像与文本两个模态特征相互感知，以实现更好的自适应零样本对齐.|<img src="./images/CALIP.png"  width="1280px"/>| [[Github](https://github.com/ZiyuGuo99/CALIP)] <br> [[Paper](https://arxiv.org/pdf/2209.14169)]  |
| [![Star](https://img.shields.io/github/stars/KaiyangZhou/CoOp.svg?style=social&label=Star)](https://github.com/KaiyangZhou/CoOp)   <br> **CoOp: Learning to Prompt for Vision-Language Models** <br>| 受NLP领域prompt learning的启发，文章提出了Context Optimization(CoOp)，用于将类CLIP式的视觉语言模型迁移到下游图像识别任务.具体而言，CoOp将预训练模型参数freeze，使用可学习向量对提示的上下文单词进行建模.作者在11个下游任务上验证CoOp的有效性，结果显示CoOp的性能明显好于原始预训练模型如CLIP.|<img src="./images/CoOp.png"  width="1280px"/>| [[Github](https://github.com/KaiyangZhou/CoOp)] <br> [[Paper](https://arxiv.org/pdf/2109.01134)] |
| [![Star](https://img.shields.io/github/stars/KaiyangZhou/CoOp.svg?style=social&label=Star)](https://github.com/KaiyangZhou/CoOp)   <br> **CoCoOp: Conditional Prompt Learning for Vision-Language Models** <br>|针对CoOp泛化性差的问题，即:学习到的上下文对数据集中unseen classes的泛化性不好.文章提出Conditional Context Optimization (CoCoOp)，在CoOp基础上，引入一个轻量级的网络，名为Meta-Net:为每张图像生成input-conditional tokens. input-conditional tokens与 CoOp中的learnable vectors叠加，共同参与训练.大量实验表明，对于unseen classes，CoCoOp 比 CoOp 的泛化能力要好得多，甚至显示出超越单个数据集的可迁移性， 并产生更强的领域泛化性能 |<img src="./images/CoCoOp.png"  width="1280px"/>| [[Github](https://github.com/KaiyangZhou/CoOp)] <br> [[Paper](https://arxiv.org/pdf/2203.05557)] |


## Retrieval

## Captioning

| Title | Abstract | Intro | Useful Links |
|:----| :---:| :----: | :---:|
|2021|
| [![Star](https://img.shields.io/github/stars/rmokady/CLIP_prefix_caption.svg?style=social&label=Star)](https://github.com/rmokady/CLIP_prefix_caption) <br> **ClipCap: CLIP Prefix for Image Captioning** <br>| 作者提出了CLIPCap模型来生成image captions.具体而言，借助CLIP提取图像embeadding，训练一个mapping network，为每一个caption生成前缀.直接和caption embedding 做结合（concatenation），形成新的embedding，送入GPT-2生成captions. |<img src="./images/ClipCap.png"  width="1280px"/>| [[Github](https://github.com/rmokady/CLIP_prefix_caption)] <br> [[Paper](https://arxiv.org/pdf/2111.09734)] |

##


