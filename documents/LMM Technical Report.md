This technical report is to provide an overview of the large multimodal models. The report includes the following sections. 1) What is Large multimodel model 2) Why we are interested in LMM 3) LMM Architeture, 4) LMM Training Strategy and Data, and 5)LMM for Document Understanding. At the end of the this technical report, we will also include some highlights of a few recent LMM models.

# 1. What is LMM?

Large Multi-Modal Model <-> Multi-Modal Large Language Model <-> Vision Language Model


Formally, it (Multimodal large language model) refers to the LLM-based model with the ability to receive, reason, and output with multimodal information.

LMM manifests two representative traits compared with the traditional counterparts: (1) LMM is based on LLM with billion scale parameters, which is not available in previous models. (2)LMM uses new training paradigms to unleash its full potential, such as multimodal instruction tuning to encourage the model to follow new instructions. 

# 2. Why we are interested in LMM?

It is to explore the art-of-possible, more specifically, to explore if and how LMM can make the document understanding solution more effective, more efficient and more scalable. 

## 2.1 Potential Improvements to the current DU pipeline 
We are looking at a few aspects of the DU pipeline for improvement. 1) The DU pipeline is very complex, consisting of 8 steps and more than 10 machine learning models. The complexity makes it very difficult to scale, not only scale up but also scale out. 2) The team has put a lot of effort to improve the efficiency of the DU pipeline in the last a few months, including trying faster models, parallel computing, revising post-processing of information extraction, in addition to the effort of having new hardware and platform. 3) The model performance is still to be improved across the DU pipeline. It will require substantial amount of effort to get the training data, building and validating multiple models in the DU pipeline. 

## 2.2 LLM and LMM
LLM has revolutionised the field of artificial intelligence, enabling natural language processing teask that were previously through exclusive to humans. The emergence of GPT-3 brought a signficant leap in capabilities, particularly in few shot and zero-shot learning, highlighing the immence potential of LLMs. This promis was further realised with the advancements of ChatGPT and GPT-4. The progress in the field has been further accelerated by the emergence of open-source LLMs including the LLaMA series, Vicuna, Qwen etc.



## 2.2 Document understanding solution - Two Steps VS OCR Free
Extracting key information from a variety of sources, including documents like tables, forms, and invoices, as well as text in the wild is crucial for industries and acadmeic research, aiming to automate the refine document-based and scene-text workflows. This field requires text detection and recognition in both document images and real-world scenes, language comprehension, and the integration of vision and language.

Many early methods attempt to address the task using a two-stage approach: 1)Detect and recognise the text using external systems; 2)Document understanding based on the fusion of text results and images. However, the individual step of text reading in the processing pipeline may lead to the accumulation of errors. Moreover, relying on off-the-shelf OCR Models/APIs (OCR-Models) introduces additional engineering complexity, limits the connection between the text and its surrounding context, and can potentially increase computational costs. 

# 2. LMM Architeture
A typical LMM can be abstracted into three modules, i.e., a pre-trained modality encoder, a pre-trained LLM, and a modality interface to connect them. 

The typical LMM architechture is as follows (please refer to the repository wiki to know more about inserting images in github):

<!-- ![architecture](pictures/architecture.png =500x200) -->
<p align="center">
  <img src="pictures/architecture.png" width="750" />
</p>

## 2.1 Modality encoder
The encoders compress raw information, such as images or audio, into a more compact representation. Rather than training from scratch, a common approach is to use a pre-trained encoder that has been aligned to other modalities.

Rather than training from scratch, a common approach is to use a pretrained encoder that has been aligned to other modalities. 

For exaple, CLIP [[1]](#1) incorporates a visual encoder semantically aligned with the text through large-scale pre-training on image-text paris. Therefore, it is easier to use such initially pre-aligned encoders to align with LLMs through alignment pre-training. Some commonly used image encoders are listed in Table 1:

Table 1: Commonly Used Image Encoders

|   Variants | pretaining corpus | Resolution | Samples(B) | Parameter Size (M)    |
| :--- | :--- | :---: | :---: | ---: |
|OpenCLIP-ConvNext-L|LAION-2B| 320 | 29 | 197.4|
|CLIP-ViT-L/14 | OpenAI's WIT| 224/336| 13 | 304.0|
|EVA-CLIP-ViT-G/14 |LAION-2B, COYO-700M | 224| 11 | 1000.0 |
|OpenCLIP-ViT-G/14 | LAION-2B | 224 | 34 | 1012.7 |
|OpenCLIP-ViT-bigG/14 | LAION-2B | 224 | 34 | 1844.9 |


Notably, many works have empirically verified that using higher resolution can achieve remarkable performance gains. In contrast, parameter size and training data compostion are of less importance compared with input resolution, found by empirical studies [[2]](#2). 

Please note: this is contradictory with the claim made by the InternV model in which the authors claims a large image encoder generates far more better results.

## 2.2 Pre-trained LLM

Instead of training an LLM from scratch, it is more efficient and practical to start with a pre-trained one. LLaMA series and Vicuna family are representative open-srouced LLMs that have attracted much attention. 

It should be noted that scaling up the parameter size of LLMs also brings additional gains, similar to the case of increasing input resolution. 

Recently, explorations of Mixture of Experts (MoE) architecture for LLMs have garnered rising attention. Compared with dense models, the sparse architecture enables scaling up total parameter size without increasing computational cost, by selective activation of parameters. Empirically, MM1 and MoE-LLaVA find that MoE implementation achieves better performance than the dense counterpart on almost all the benchmarks.


## 2.3 Modality Interface

Since LLMs can only perceive text, bridging the gap between natural language and other modalities is necessary. However, it would be costly to train a large multimodal model in an end-to-end manner. A More practical way is to introduce a learnable connector between the pre-trained visual encoder and LLM. The other approach is to translate images into languages with the help of expert models, and then send the language to LLM.

**Learnable Connector**. It is responsible for bridging the gap between different modalities. Specifically, the module projects information into the space that LLM can understand efficiently. Based on how multimodal information is fused, there are broadly two ways to implement such interfaces, *i.e.*, token-level and feature-level.

For token-level fusion, features output from encoder are transformed into tokens and concatenated with text tokens before being sent into LLMs. A common and feasible solution is to leverage a group of learnable query tokens to extract information in a query-based manner. Such Q-Former style approaches compress visual tokens into a smaller number of representation vectors. In contrast, some methods simply use a MLP-based interface to bridge the modality gap. For example, LLaVA [[3]](#3) adopts one/two linear MLP to project visual tokens and align the feature dimension with word embeddings.

On a related note, MM1 has ablated on design choices on the connector and found that for token-level fursion, the type of modality adapter is far less important than the number of visual tokens and input resolution.

As another line, feature-level fusion inserts extra modules that enable deep interaction and fusion between text features and visual features. 

Empirially reveal that the token-level fusion variant performs better in terms of VQA benchmarks. Rgarding the performance gap, the authors suggest that cross-attention models might require a more complicated hyper-parameter searching process to achieve comparable performance.


**Expert Model**. Apart from the learnable interface, using expert models, such as an image captioning model, is also a feaible way to bridge the modality gap. The basic idea is to convert multimodal inputs into languages without training. In this way, LLMs can understand multimodality by the converted languages. Though using expert models is straightforward, it may not be as flexible as adopting a learnable interface. The conversion of foreign modalities into text would cause information loss.


## 2.4 Some models using multiple resolution of images for pretraining
In these models, the visual encoder is no longer a CLIP as CLIP needs the image and text pair as training data. Instead these models using a pure visual encoder without text information. For example, InternVL used a a so-called dynamic high-resolution strategy to train a strong vision encoder named Intern ViT-6B-448px-V1.5. TextMonkey used a strategy including three steps: 1)shifted Window Attention; 2)Image Resampler; and 3)token resampler. In TextMonkey, the positional cues of the ansers were extracted and integrated into the answers themselves. Because of this strategy, it has a certain level of capacility of grounding, i.e., identifying the position of the information being extracted from the images. UReader also trained a visual encoder before passing the information to the LLM for instruction tuning. It also includes some grid information which makes it possible to present certain level of grounding capability.  


# 3. LMM Training Strategy and Data
A full-fledged LMM undergoes three stages of training, i.e. pre-training, instruction-tuning, and alignment tuning. Each phase of training requires different types of data and fulfills different objects.

## 3.1 Pre-training

### 3.1.1 Training Detail
As the first training stage, pre-training mainly aims to align different modalities and learn multimodal world knowledge. Pre-training stage generally entails large-scale text-paired data, e.g., caption data. Typically, the caption pairs describe images/audio/videos in natural language sentences. A common approach for pre-training is to keep pre-trained modules (e.g. visual encoders and LLMs) frozen and train a learnable interface. 

### 3.1.2 Data
Pretraining data mainly serve two purposes, i.e. (1) aligning different modalities and (2) providing world knowledge.

## 3.2 Instruction-Tuning
### 3.2.1 Introduction

Instruction refers to the description of tasks. Intuitively, instruction tuning aims to teach models to better understand the instructions from users and fulfill the demanded tasks. The comparisons between instruction tuning and related typical learning paradigms are illustrated in Fig. 3. The supervised fine-tuning approach usually requires a large amount of task-specific data to train a task-specific model. The prompting approach reduces the reliance on large-scale data and can fulfill a specialized task via prompt engineering. In such a case, though the few-shot performance has been improved, the zero-shot performance is still quite
average. Differently, instruction tuning learns how to generalize to unseen tasks rather than fitting specific tasks like the two counterparts. Moreover, instruction tuning is highly related to multi-task prompting.

### 3.2.2 Training Detail
A multimodal instruction sample often includes an optional instruction and an input-output pair. The instruction is typically a natural language sentence describing the task, such as, “Describe the image in detail.” The input can be an image-text pair like the VQA task or only an image like the image caption task. The output is the answer to the instruction conditioned on the input.




# 4. LMM for Document Understanding

# 5. LMM Demos




# Selected LMM Papers

## Paper: LayoutLLM: Layout Instruction Tuning with Large Language Models for Deocument Understanding [[4]](#4) 

The architechture and training paradigms are show in the figure below.

<!-- ![LayoutLLM](pictures/LayoutLLM.png) -->

<p align="center">
  <img src="pictures/LayoutLLM.png" width="750" />
</p>

The overall architecture is as follow:

<!-- ![LayoutLLM_overall](pictures/LayoutLLM_overall_architecture.png) -->

<p align="center">
  <img src="pictures/LayoutLLM_overall_architecture.png" width="750" />
</p>


## Paper: InternVL: Scaling up Vision Foundation Modesl and Aligning for Generic Visual-Linguistic Tasks [[5]](#5) 



The training strategy of InternVL consists of three progressive stages, including vision-language
contrastive training, vision-language generative training, and supervised fine-tuning. These stages effectively leverage public data from diverse sources, ranging from noisy image-text pairs on the web to high-quality caption, VQA, and multi-modal dialogue datasets. The working flow is show in the following figure:

<!-- ![training_strategy](pictures/Internvl_training_strategy.png) -->

<p align="center">
  <img src="pictures/Internvl_training_strategy.png" width="1000" />
</p>


## Paper: MM1: Methods, Analysis & Insights from Multimodal LLM Pre-training [[6]](#6) 

In particular, we study the importance of various architecture components and data choices. Through
careful and comprehensive ablations of the image encoder, the vision
language connector, and various pre-training data choices, we identified several crucial design lessons. For example, we demonstrate that for
large-scale multimodal pre-training using a careful mix of image-caption,
interleaved image-text, and text-only data is crucial for achieving stateof-the-art (SOTA) few-shot results across multiple benchmarks, compared to other published multimodal pre-training results. Further, we
show that the image encoder together with image resolution and the image token count has substantial impact, while the vision-language connector design is of comparatively negligible importance



## References
<a id="1">[1]</a>
A.Radford, J.W.Kim, C.Hallacy, A.Ramesh, G.Goh, S.Agarwal, G.Sastry, A.Askell, P.Mishkin, J.Clark et al.,
Learning transferable visual models from natural language supervisions.
in ICML, 2021

<a id="2">[2]</a>
B. McKinzie, Z. Gan, J.-P Fauconnier, S. Dodge, B. Zhang, P. Dufter, D. Shah, X. Du, F. Peng, F. Weers et al.,
Mm1: Methods, analysis & insights from Multimodal LLM pre-training,
arXiv:2403.09611v4

<a id="3">[3]</a>
H.Liu, C.Li, Q.Wu, and Y.J.Lee, 
Visual instruction tuning,
NeruIPS 2023

<a id="4">[4]</a>
Chuwei Luo and Yufan Shen and Zhaoqing Zhu and Qi Zheng and Zhi Yu and Cong Yao, 
LayoutLLM: Layout Instruction Tuning with Large Language Models for Document Understanding,
CVPR 2024

<a id="5">[5]</a>
Chen, Zhe and Wang, Weiyun and Tian, Hao and Ye, Shenglong and Gao, Zhangwei and Cui, Erfei and Tong, Wenwen and Hu, Kongzhi and Luo, Jiapeng and Ma, Zheng and others
How Far Are We to GPT-4V? Closing the Gap to Commercial Multimodal Models with Open-Source Suites
arXiv:2404.16821, 2024

<a id="6">[6]</a>
Brandon McKinzie and Zhe Gan and Jean-Philippe Fauconnier Biard and Sam Dodge and Philipp Dufter and Bowen Zhang and Dhruti Shah and Xianzhi Du and Futang Peng and Haotian Zhang and Floris Weers and Anton Belyi and Karanjeet Singh and Doug Kang and Ankur Jain and Hongyu He and Max Schwarzer and Tom Gunter and Xiang Kong and Aonan Zhang and Jianyu Wang and Chong Wang and Nan Du and Tao Lei and Sam Wiseman and Mark Lee and Zirui Wang and Ruoming Pang and Peter Grasch and Alexander Toshev and Yinfei Yang
MM1: Methods, Analysis & Insights from Multimodal LLM Pre-training,
arXiv:2403.09611v4, 2024
