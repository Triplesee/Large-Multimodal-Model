---
marp: true
---


# Multi-modal Large Language Model - An Intro

---
# **overview**

## 1. What is MLLM?

## 2. Why MLLM?

## 3. MLLM Architeture

## 4. MLLM Training Strategy and Data

## 5. MLLM for Document Understanding
---
# The motivation 

The research into LMM is to see if we can find alternative approaches 
for document understanding. Currently the document understanding pipeline 
is doing a good job. However, it is fairly complex with 
many steps. The training needs a lot of data and time which is very expensive.

On the other hand, MLLM attracted a lot of attention because it 

You might have watched a nice video a few weeks ago called "Multimodal Generative AI Demystified". The Multimodal models mentioned in that video includes many types of input and many types of output. 

This presentation as shown in the title, is to explore the impact of LMM to document understanding, we will be focusing on the multimodal models with vision (or Image) input. And it doesn't really need to generate images either as the key requirements of document understanding is to use real images/document to support decision making. 

---

# What is MLLM?

## Multimodal Large Language Model <-> Large Multimodal Model

## LLM-based model with the ability to **receive**, **reason**, and **output** with multimodal information. 

## MLLM manifests two representative traits: 
(1) MLLM is based on LLM with billion scale parameters, which is not available in previous models. 

(2) MLLM uses new training paradigms to unleash its full potential, such as multimodal instruction tuning to encourage the model to follow new instructions. 

---
# 2. MLLM Architecture

A typical MLLM can be abstracted into three modules, i.e., 
 1) a pre-trained modality encoder, 
 2) a pre-trained LLM, and 
 3) a modality interface to connect them. 

The typical MLLM architechture is as follows

![architecture](pictures/architecture.png)

---


# **overview**

### 1. LMM Introduction

### 2. LMM Architeture

### 3. LMM Training Strategy and Data

### 4. A few LMM

---
# Recommendation

When building Vector Databases, shall we also consider the vectors for multimodal data such as images particularly (scanned or photo) documents?
---

# Further resources

## Youtube Video:
* [Mastering Multimodal Models: Exploring Idefics2](https://www.youtube.com/watch?v=DrdlIxOC5ig)
* [Multimodal Generative AI Demystified](https://www.youtube.com/watch?v=8V2cUcuasYQ&t=8s)
## Website:
* [InternVL](https://github.com/OpenGVLab/InternVL)
* [LLaVA](https://github.com/haotian-liu/LLaVA)
* [TextMonkey](https://github.com/Yuliang-Liu/Monkey)
* [Awesome-Multimodal-Large-Language-Models](https://github.com/BradyFU/Awesome-Multimodal-Large-Language-Models)
* [Awesome-Document-Understanding](https://github.com/harrytea/Awesome-Document-Understanding)
---
# End of slide deck
