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

## Recommended reading material
