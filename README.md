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

Although the stated objective is to use generative tecniques to address 3D cloud reconstruction, depending on the interests and skills of the team we may explore several aspects of 3D cloud reconstruction:
1. ML aspects of reconstructing 3D cloud fields such as:
    - producing uncertainty estimation
    - generative techniques
    - self-supervision techniques
    - sensor independence tecniques
    - others
2. Downstream modeling. Use available 3D cloud models for modeling downstream cloud related phenomena such as:
    - Tropical cyclone intensification
    - Precipitation 

### 🗂️ Data

 The dataset is composed of three sub-datasets:
1. A pretraining, geostationary satellite imagery dataset. 
2. A finetuning paired dataset with geostationary image and cloudsat vertical profile pairs. 
4. A Tropical Cyclone paired dataset with geostationary image and cloudsat vertical profiles along selected TC tracks.

The full dataset is quite large so you will be provided a mini-version for development during the week. The full dataset is hosted at [data.coop](https://data.coop/en/) and can be accessed through notebooks, including some in this repo, but for quicker ML development we will provide the reduced version. 

| dataset     | version | size   |
|-------------|---------|--------|
| pretraining | full    | 453 GB |
| finetuning  | full    | 324 GB |
| TC          | full    | 1 GB   |
| pretraining | dev     | 20 GB  |
| finetuning  | dev     | 10 GB  |
| TC          | dev     | 1 GB   |


 For more details look at first reference from recommended reading material and go [here](/README_data.md)



### 🚀 Getting Started (pre-challenge prep)


🅐  Set up python environment

🅑 Explore data
- Explore the pretraining, geostationary imagery dataset with [this notebook](https://github.com/WinterSchool2026/ch10-3d-hurricane-generation/blob/main/notebooks/00_data_geostationary.ipynb)
- Explore the finetuning, geostatinary image - cloudsat vertical profile, dataset.
    

🅒 Explore minimalist 3D cloud ML models

- Explore a minimal U-net autoencoder based pre-training model with [this notebook](https://github.com/WinterSchool2026/ch10-3d-hurricane-generation/blob/main/notebooks/01_model_pretraining.ipynb)
- Explore a minimal U-net regression model 


### 🧩  Challenge plan 

1️⃣ Challenge presentation. - Monday 11:45-12:30
- Brief presentation on dataset and previous work
- Discussion and Questions

2️⃣ Session 1 - Monday 14:45-17:45
- Miro board non-verbal session to agree on direction and design workflow
- Pair coding 

3️⃣ Session 2 - Wednesday 14:45-17:45
- Pair coding

4️⃣ Session 3 - Thursday 14:45-17:45
- Presentation planning
- Pair coding & presentation 

5️⃣ Presentation - Friday 14-16:15
- present progress