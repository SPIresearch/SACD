# SACD
[PyTorch](https://pytorch.org/)  implementation on: Seeing your sleep stage: cross-modal distillation from EEG to infrared video.                                                                                    
      (A cross-modal distillation method based on infrared video and EEG signals)


## Introductio
We propose a novel cross-modal methodology (SACD) to solve the previous barriers, enabling point-of-care sleep stage monitoring at home.

To enable the developments of point-of-care healthcare research and distillation methods from clinical to visual modality, to our best knowledge, we are the first to collect a large-scale cross-modal distillation dataset, namely $S^3VE$.
<p align="center">
<img src="https://github.com/SPIresearch/SACD/blob/main/SACD/OVERVIEW.png" width="75%">
</p>

## Getting Started
### Prerequisites:
- python >= 3.6.10 
- pytorch >= 1.1.0
- FFmpeg, FFprobe
- Download $S^3VE$ datasets:
[EGG](https://pan.baidu.com/s/1mhRdYQEzTqR9rLwW4OZv6Q),
[Infrared video features]( https://pan.baidu.com/s/1yuUIXqNoZqXPAB_8uIO0ag),
[labels](https://pan.baidu.com/s/1GvBR3dLqj6KRmpG1YpVyDQ)
- Download pretrained weights:
We also provide the pre-trained weights of the IR video encoder and the weights after our method distillation. [link](https://pan.baidu.com/s/1ryaxMGupD-wu2I_bT7iQNg )


Note：If you need our dataset for relevant research, please send us an email with your Institution, Email and Use Purpose！we will provide you with the Baidu Cloud extraction code after review. Our email address is: 715129324@qq.com

### Data Preparation on S^3VE (example):
