For more details look at first reference from recommended reading material and go [here]()

#### Pretraining/Unsupervised dataset

The dataset is composed of 512 x 512 pixel images from 3 geostationary satellites:
    - GOES: 16 bands, coverage of Americas
    - Himawari: 16 bands, coverage of Asia and Oceania 
    - MSG: 11 bands, coverage of Europe and Africa

#### Finetuning/Supervised dataset

Dataset is composed of paired geostationary images, cloudsat vertical profile observations: 
Each pair is composed of a co-located geostationary image and vertical profile pair

The geostationary image is composed of:
- 256 by 256 pixels
- 18-23 “channels”
- 11-16 spectral channels
- 4 channels describing sensor geometry:  satellite viewing angle (zenith
and azimuth), and solar angle (zenith and azimuth) 
- 2 channels describing time of measurement: fraction of year and fraction of day.
cloudsat overhead mask showing the pixels where the path along the image where the cloudsat profile is available. 
- cloudsat overpass mask indicating pixels for which a vertical profile is available.

The cloudsat vertical profile is composed of:
- W pixels along the image (variable length 256-512) and 125 height levels. 
- 3 variables: Radar reflectivity, Effective droplet radius and Ice Water Content