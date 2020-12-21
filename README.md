## Trash Classification
The repo contains code for a SSD-Sinlge Shot Detector to detect trash in an image. It takes input as an image and outputs bounding boxes over all the instance of trash in the image.
## Motivation
This project was done as a part of the CS344/CS386 Course Project, under the guidance of Dr. Clint P. George.

  -->
 
## Tech
<b>Built with</b>
- Python

## Sources/Refrences
- The repository https://github.com/pierluigiferrari/ssd_keras is a main source of code.
- The SSD paper : https://arxiv.org/pdf/1512.02325.pdf
- The dataset is from : https://github.com/pedropro/TACO.

For details on the dataset : http://tacodataset.org/

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
Two models are present in the model_weights folder 
Model: Uses mapping of 10 categories 
Model_v2: Uses mapping of 19 categories
Train:
```
python3 run.py train --dataset=../TACO/data --annot_train=map10_without_batch10_15_train.json --annot_val=map10_without_batch10_15_val.json

```

```
python3 run.py train --dataset=../TACO/data --config=config_v2.json --model=model_weights/Model_v2.h5 --annot_train=custom_map_train_lskw_rm10_15.json --annot_val=custom_map_val_lskw.json


```

Evaluate: In this just give a single image path

```
python3 run.py evaluate --evaluate_img=../TACO/data/batch_12/000044.jpg   --model=model_weights/Model.h5 --config=config.json
```

```
python3 run.py evaluate --evaluate_img=../TACO/data/batch_12/000044.jpg --model=model_weights/Model_v2.h5 --config=config_v2.json

```
Test image paths are provided in test_samples.txt change the path of image in the above command with the image path to generate the images.

The config files contain all the required parameters as descirbed in config_details.csv

The model sets provided are :

For Model.h5 and config.json
- SSD/map10_without_batch10_15_train.json
- SSD/map10_without_batch10_15_val.json


For Model_v2.h5 and config_v2.json
- SSD/custom_map_train_lskw_rm10_15.json
- SSD/custom_map_val_lskw.json

For details on running the run.py with different arguments do 
```
python3 run.py -h
```

## Team
Ankit, Devyani, Siddharth (Indian Institute of Technology Goa)
