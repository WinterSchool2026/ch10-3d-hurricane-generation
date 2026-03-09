
## 3D Clouds data

#### Pretraining/Unsupervised dataset

The dataset is composed of 512 x 512 pixel images from 3 geostationary satellites:

    - GOES: 16 spectral bands, coverage of Americas
    - Himawari: 16 spectral bands, coverage of Asia and Oceania 
    - MSG: 11 spectral bands, coverage of Europe and Africa

There is metadata available for each geostationary image. 

#### Finetuning/Supervised dataset

Dataset is composed of paired geostationary images, cloudsat vertical profile observations: 
Each pair is composed of a co-located geostationary image and vertical profile pair

Each geostationary image is composed of:

- 256 by 256 pixels
- 18-23 “channels”
- 11-16 spectral channels
- Metadata for the geostationary image including:
    - sensor geometry  satellite viewing angle (zenith and azimuth), and solar angle (zenith and azimuth) 
    - time of measurement
    - geographical coordinates
- cloudsat overpass mask indicating pixels for which a vertical profile is available.
 
The cloudsat vertical profile is composed of:
- W pixels along the image (variable length 256-512) and 125 height levels. 
- 3 variables: Radar reflectivity, Effective droplet radius and Ice Water Content
- Metadata for the cloudsat profile


