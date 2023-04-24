# Fine-grained-Sequence-to-Sequence-Lip-Reading-Based-on-Self-Attention-and-Self-Distillation

The PyTorch implementation of the paper ***Fine-grained-Sequence-to-Sequence-Lip-Reading-Based-on-Self-Attention-and-Self-Distillation***


## Introduction

**In this paper, we propose a fine-grained method based on self-attention and self-distillation. The whole model mainly includes the CNN front-end, pixel-wise learning, temporal learning, and decoder. Specifically, we apply the CNN front-end to capture shallow spatial features inside the image sequence, and employ the Resformer module including self-attention to learn the global spatial correlation between pixels, namely, pixel-wise learning. Then, the encoder is utilized to learn the temporal features, namely, temporal learning. Finally, the decoder decodes visual information to realize text prediction. Furthermore, we utilize self-distillation to further promote model performance. Experiments show that the WER metric of our model decreases to 1.77%, 14.75%, and 54.60% for seq2seq tasks on GRID, LRW, and LRW-1000, respectively, which has significant improvement compared with existing methods.**


![模型图](https://github.com/shibo0/Fine-grained-Sequence-to-Sequence-Lip-Reading-Based-on-Self-Attention-and-Self-Distillation/blob/main/imgs/model5.jpg)

## Reference

```
Junxiao XUE, Shibo HUANG, Huawei SONG, Lei SHI. Fine-grained sequence-to-sequence lip reading based on self-attention and self-distillation. Front. Comput. Sci., 2023, 17(6): 176344 https://doi.org/10.1007/s11704-023-2230-x
```
