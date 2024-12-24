# ChangeDiff
(AAAI-2025) ChangeDiff: A Multi-Temporal Change Detection Data Generator with Flexible Text Prompts via Diffusion Model


This is a [pytorch](http://pytorch.org/) implementation of our paper [ChangeDiff](https://arxiv.org/pdf/2412.15541).
(AAAI-2025)


## :speech_balloon:  Multi-temporal semantic change synthetic data
It is trained on the sparsely labeled semantic change detection SECOND (Yang et al. 2021) dataset.
![](./images/fig1.jpg)

## :speech_balloon: ChangeDiff Pipeline
![](./images/pipeline.jpg)

## :speech_balloon: ChangeDiff Training, sampling.
Please find the corresponding training, sampling and testing scripts under the corresponding files.



### Acknowledge
Some codes are adapted from [FreestyleNet](https://github.com/essunny310/FreestyleNet),  [TokenCompose](https://github.com/mlpc-ucsd/TokenCompose) and [SCanNet](https://github.com/ggsDing/SCanNet). We thank them for their excellent projects.


### Citation
If you find this code useful please consider citing
```
@misc{zang2024ChangeDiff,
      title={ChangeDiff: A Multi-Temporal Change Detection Data Generator with Flexible Text Prompts via Diffusion Model}, 
      author={Qi Zang and Jiayi Yang and Shuang Wang and Dong Zhao and Wenjun Yi and Zhun Zhong},
      year={2024},
      eprint={2412.15541},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2412.15541}, 
}
```




