## Trash Classification
The repo contains code for a SSD-Sinlge Shot Detector to detect trash in an image. It takes input as an image and outputs bounding boxes over all the instance of trash in the image.
## Motivation
This project was done as a part of the CS344/CS386 Course Project, under the guidance of Dr. Clint P. George.
<!-- 
## Build status
Build status of continus integration i.e. travis, appveyor etc. Ex. - 

[![Build Status](https://travis-ci.org/akashnimare/foco.svg?branch=master)](https://travis-ci.org/akashnimare/foco)
[![Windows Build Status](https://ci.appveyor.com/api/projects/status/github/akashnimare/foco?branch=master&svg=true)](https://ci.appveyor.com/project/akashnimare/foco/branch/master) -->

<!-- ## Code style
If you're using any code style like xo, standard etc. That will help others while contributing to your project. Ex. -

[![js-standard-style](https://img.shields.io/badge/code%20style-standard-brightgreen.svg?style=flat)](https://github.com/feross/standard)
  -->
 
## Tech
<b>Built with</b>
- Python

The repository ?? is a main source of code.


## Libraries
The libraries used in building the project are listed here :
- Tensorflow 2.x
- Keras 2.x
- sklearn, imageio,open-cv

## Usage
Clone the repo and then in the TACO/Data directory run the following command:
All dependencies are provided in the requirement.txt file
```
python3 download.py
```
This will download the TACO dataset for trash detection into the proper directory sa required to run other code.

From the SSD folder run the following commands depending on usage:
Train:
```
python3 run.py train --dataset=../TACO/data --pretrained_weights=False
```

Evaluate: In this just give a single image path
```
python run.py evaluate --evaluate_img=../TACO/data/batch_2/000000.JPG

```

## Team
Ankit, Devyani, Siddharth (Indian Institute of Technology Goa)
