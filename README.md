# Generating 3D video of hurricanes

## Description

Use deep learning to obtain 3D maps of hurricanes from multispectral 2d
geostationary imagery. The goal is to use 2d satellite imagery as input and vertical profiles of
clouds, consisting of radar/lidar from cloudsat, as output. Since cloudsat measurements are
sparse in space and time (narrow swath with approximately monthly revisit time) , estimating
vertical profiles from geostationary imagery allows for 15 minute cadence 3d maps.
Although infra-red channels and spatial information contained in the 2d imagery include
information on the vertical dimension, as shown by success of deep learning models in this
task, not all the vertical information is available. This means that for a given 2d image it is
more appropriate to produce a distribution of possible 3d cloud maps. The goal is to use
generative techniques, such as latent diffusion, to address this challenge.

The dataset we will use was constructed as part of the [Earth System Lab 2025 (ESL) - 3D Clouds for climate
extemes challenge](https://eslab.ai/fdl-esl-2025).


### 📚 Recommended reading material
- [ESL 2025 paper](https://arxiv.org/abs/2511.04773) 

### 🎯 Challenge Objectives

### 🗂️ Data

 The dataset is composed of two sub-datasets:
- A pretraining, geostationary satillite imagery dataset. 
- A finetuning dataset with geostationary image and cloudsat vertical profile pairs. 
 
 For more details look at first reference from recommended reading material and go [here](/README_data.md)


### 🚀 Getting Started (pre-challeng prep)


1️⃣ Set up python environment

2️⃣ Explore data
    
    - Explore the pretraining, geostationary imagery dataset with [this notebook](https://github.com/WinterSchool2026/ch10-3d-hurricane-generation/blob/main/notebooks/00_data_geostationary.ipynb)
    - Explore the finetuning, geostatinary image - cloudsat vertical profile, dataset.
    

3️⃣ Explore minimalist 3D cloud ML models

- Explore a minimal U-net autoencoder based pre-training model with [this notebook](https://github.com/WinterSchool2026/ch10-3d-hurricane-generation/blob/main/notebooks/01_model_pretraining.ipynb)
- Explore a minimal U-net regression model 

4️⃣

5️⃣

6️⃣

7️⃣

8️⃣

### 🚀 Challenge plan 

1️⃣ Challenge presentation

2️⃣ Session 1

3️⃣ Session 2

4️⃣ Session 3

5️⃣ Presentation: present progress