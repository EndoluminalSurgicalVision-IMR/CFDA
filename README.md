# CFDA

<div align=center><img src="figs/main%20framework.png"></div>

[**_Collaborative Feature Disentanglement and Augmentation for Pulmonary Airway Tree Modeling of COVID-19 CTs_**]()

> By Minghui Zhang, Hanxiao Zhang, Guang-Zhong Yang and Yun Gu

## Introduction
This repository is for our early accepted MICCAI 2022 paper 'Collaborative Feature Disentanglement and Augmentation for Pulmonary Airway Tree Modeling of COVID-19 CTs'.

> Detailed modeling of the airway tree from CT scan is important for 3D navigation involved in endobronchial intervention including for those patients infected with the novel coronavirus. Deep learning methods have the potential for automatic airway segmentation but require large annotated datasets for training, which is difficult for a small patient population and rare cases. Due to the unique attributes of noisy COVID-19 CTs (e.g., ground-glass opacity and consolidation), vanilla 3D Convolutional Neural Networks (CNNs) trained on clean CTs are difficult to be generalized to noisy CTs. We propose a Collaborative Feature Disentanglement and Augmentation framework (CFDA) to harness the intrinsic topological knowledge of the airway tree from clean CTs incorporated with unique bias features extracted from the noisy CTs.

## Usage
You can set up the options in the **options/base_options** and modify the template in the **pipeline/pipeline_template**, 
e.g., for the training of the CFDA:

```
python pipeline_train_CFDA.py
```
