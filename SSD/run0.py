# -*- coding: utf-8 -*-
# Use the commands to make use of the code in the repository
# python3 run.py train --dataset=../TACO-master/data --pretrained_weights=False
# python run.py evaluate --evaluate_img=../TACO-master/data/batch_2/000000.JPG

# making necessary imports here
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing import image
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TerminateOnNaN, CSVLogger
from keras import backend as K
from tensorflow.keras.models import load_model
from math import ceil
import numpy as np
from matplotlib import pyplot as plt
from imageio import imread

from models.keras_ssd7_our_gpu_version import build_model
from keras_loss_function.keras_ssd_loss_yani import SSDLoss
from keras_layers.keras_layer_AnchorBoxes_v3_yani import AnchorBoxes
# from keras_layers.keras_layer_AnchorBoxes import AnchorBoxes
from keras_layers.keras_layer_DecodeDetections import DecodeDetections
from keras_layers.keras_layer_DecodeDetectionsFast import DecodeDetectionsFast

from ssd_encoder_decoder.ssd_input_encoder import SSDInputEncoder
from ssd_encoder_decoder.ssd_output_decoder import decode_detections, decode_detections_fast

from data_generator.object_detection_2d_data_generator import DataGenerator
from data_generator.object_detection_2d_misc_utils import apply_inverse_transforms
from data_generator.data_augmentation_chain_variable_input_size import DataAugmentationVariableInputSize
from data_generator.data_augmentation_chain_constant_input_size import DataAugmentationConstantInputSize
from data_generator.data_augmentation_chain_original_ssd import SSDDataAugmentation
from data_generator.object_detection_2d_geometric_ops import Resize
import json

import argparse

if __name__=='__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run SSD on TACO.')
    parser.add_argument("command", metavar="<command>",help="Opt: 'train', 'evaluate'")
    parser.add_argument('--pretrained_weights', required=False,default=False,type=bool, help="True or False")
    parser.add_argument('--model', required=False,default="model_weights/Model.h5",metavar="/path/model.h5", help="Path to weights .h5 file")
    parser.add_argument('--dataset', required=False, metavar="/path/dir", help='Directory of the dataset',default='../TACO-master/data')
    parser.add_argument('--lrate', required=False, default=0.001, type=float, help='learning rate')
    parser.add_argument('--evaluate_img',required=False,default='images/img.jpg',metavar="/path/dir", help='Directory of the dataset')
    # parser.add_argument('--class_map', required=True, metavar="/path/file.csv", help=' Target classes')
    DEFAULT_LOGS_DIR="/training_logs"
    args = parser.parse_args()
    print("Command: ", args.command)
    print("pretrained_weights: ",args.pretrained_weights)
    print("Model: ", args.model)
    print("Dataset: ", args.dataset)
    print("Logs: ", DEFAULT_LOGS_DIR)











## 1. Set the model configuration parameters in the config.json file

with open('config.json') as f:
  config = json.load(f)

img_height=config["img_height"] # Height of the input images
img_width=config["img_width"] # Width of the input images
img_channels=config["img_channels"] # Number of color channels of the input images
intensity_mean=config["intensity_mean"]#Set this to your preference (maybe `None`). The current settings transform the input pixel values to the interval `[-1,1]`.
intensity_range=config["intensity_range"] # Set this to your preference (maybe `None`). The current settings transform the input pixel values to the interval `[-1,1]`.
n_classes=config["n_classes"] # Number of positive classes

scales=config["scales"] #An explicit list of anchor box scaling factors. If this is passed, it will override `min_scale` and `max_scale`.
aspect_ratios =config["aspect_ratios"]#The list of aspect ratios for the anchor boxes

two_boxes_for_ar1=config["two_boxes_for_ar1"] #Whether or not you want to generate two anchor boxes for aspect ratio 1
steps=config["steps"]   #In case you'd like to set the step sizes for the anchor box grids manually; not recommended
offsets=config["offsets"]   #In case you'd like to set the offsets for the anchor box grids manually; not recommended
clip_boxes=config["clip_boxes"]# Whether or not to clip the anchor boxes to lie entirely within the image boundaries
variances=config["variances"]# The list of variances by which the encoded target coordinates are scaled
normalize_coords=config["normalize_coords"] #Whether or not the model is supposed to use coordinates relative to the image size
classes=config['classes']

"""
## 2. Build or load the model

You will want to execute either of the two code cells in the subsequent two sub-sections, not both.

### 2.1 Create a new model

If you want to create a new model, this is the relevant section for you. If you want to load a previously saved model, skip ahead to section 2.2.

The code cell below does the following things:
1. It calls the function `build_model()` to build the model.
2. It optionally loads some weights into the model.
3. It then compiles the model for the training. In order to do so, we're defining an optimizer (Adam) and a loss function (SSDLoss) to be passed to the `compile()` method.

`SSDLoss` is a custom Keras loss function that implements the multi-task log loss for classification and smooth L1 loss for localization. `neg_pos_ratio` and `alpha` are set as in the paper.
"""

# 1: Build the Keras model
# if args.command=='train':
K.clear_session() # Clear previous models from memory.


model = build_model(image_size=(img_height, img_width, img_channels),
                        n_classes=n_classes,
                        mode='training',
                        l2_regularization=0.0005,
                        scales=scales,
                        aspect_ratios_global=aspect_ratios,
                        aspect_ratios_per_layer=None,
                        two_boxes_for_ar1=two_boxes_for_ar1,
                        steps=steps,
                        offsets=offsets,
                        clip_boxes=clip_boxes,
                        variances=variances,
                        normalize_coords=normalize_coords,
                        subtract_mean=intensity_mean,
                        divide_by_stddev=intensity_range)

# 2: Optional: Load some weights
if not args.pretrained_weights:
    model.load_weights(args.model, by_name=True)

# 3: Instantiate an Adam optimizer and the SSD loss function and compile the model

adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

ssd_loss = SSDLoss(neg_pos_ratio=3, alpha=1.0)

model.compile(optimizer=adam, loss=ssd_loss.compute_loss)

"""### 2.2 Load a saved model

If you have previously created and saved a model and would now like to load it, simply execute the next code cell. The only thing you need to do is to set the path to the saved model HDF5 file that you would like to load.

The SSD model contains custom objects: Neither the loss function, nor the anchor box or detection decoding layer types are contained in the Keras core library, so we need to provide them to the model loader.

This next code cell assumes that you want to load a model that was created in 'training' mode. If you want to load a model that was created in 'inference' or 'inference_fast' mode, you'll have to add the `DecodeDetections` or `DecodeDetectionsFast` layer type to the `custom_objects` dictionary below.
"""

# # TODO: Set the path to the `.h5` file of the model to be loaded.

# model_path = 'ssd7_v4gpu_epoch-36_loss-4.1208_val_loss-4.2365.h5'
# ssd_loss = SSDLoss(neg_pos_ratio=3, alpha=1.0)
# K.clear_session()
# model = load_model(model_path, custom_objects={'AnchorBoxes': AnchorBoxes,
#                                                'compute_loss': ssd_loss.compute_loss})

# a model was given then the pretrained wts must be false
if args.command=="evaluate":
    K.clear_session() # Clear previous models from memory.


    model = build_model(image_size=(img_height, img_width, img_channels),
                            n_classes=n_classes,
                            mode='inference',
                            l2_regularization=0.0005,
                            scales=scales,
                            aspect_ratios_global=aspect_ratios,
                            aspect_ratios_per_layer=None,
                            two_boxes_for_ar1=two_boxes_for_ar1,
                            steps=steps,
                            offsets=offsets,
                            clip_boxes=clip_boxes,
                            variances=variances,
                            normalize_coords=normalize_coords,
                            subtract_mean=intensity_mean,
                            divide_by_stddev=intensity_range)
    model_path = args.model

    # We need to create an SSDLoss object in order to pass that to the model loader.
    ssd_loss = SSDLoss(neg_pos_ratio=3, n_neg_min=0, alpha=1.0)

    K.clear_session() # Clear previous models from memory.

    model = load_model(model_path, custom_objects={'AnchorBoxes': AnchorBoxes,
                                               'compute_loss': ssd_loss.compute_loss})

"""## 3. Set up the data generators for the training

The code cells below set up data generators for the training and validation datasets to train the model. You will have to set the file paths to your dataset. Depending on the annotations format of your dataset, you might also have to switch from the CSV parser to the XML or JSON parser, or you might have to write a new parser method in the `DataGenerator` class that can handle whatever format your annotations are in. The [README](https://github.com/pierluigiferrari/ssd_keras/blob/master/README.md) of this repository provides a summary of the design of the `DataGenerator`, which should help you in case you need to write a new parser or adapt one of the existing parsers to your needs.

Note that the generator provides two options to speed up the training. By default, it loads the individual images for a batch from disk. This has two disadvantages. First, for compressed image formats like JPG, this is a huge computational waste, because every image needs to be decompressed again and again every time it is being loaded. Second, the images on disk are likely not stored in a contiguous block of memory, which may also slow down the loading process. The first option that `DataGenerator` provides to deal with this is to load the entire dataset into memory, which reduces the access time for any image to a negligible amount, but of course this is only an option if you have enough free memory to hold the whole dataset. As a second option, `DataGenerator` provides the possibility to convert the dataset into a single HDF5 file. This HDF5 file stores the images as uncompressed arrays in a contiguous block of memory, which dramatically speeds up the loading time. It's not as good as having the images in memory, but it's a lot better than the default option of loading them from their compressed JPG state every time they are needed. Of course such an HDF5 dataset may require significantly more disk space than the compressed images. You can later load these HDF5 datasets directly in the constructor.

Set the batch size to to your preference and to what your GPU memory allows, it's not the most important hyperparameter. The Caffe implementation uses a batch size of 32, but smaller batch sizes work fine, too.

The `DataGenerator` itself is fairly generic. I doesn't contain any data augmentation or bounding box encoding logic. Instead, you pass a list of image transformations and an encoder for the bounding boxes in the `transformations` and `label_encoder` arguments of the data generator's `generate()` method, and the data generator will then apply those given transformations and the encoding to the data. Everything here is preset already, but if you'd like to learn more about the data generator and its data augmentation capabilities, take a look at the detailed tutorial in [this](https://github.com/pierluigiferrari/data_generator_object_detection_2d) repository.

The image processing chain defined further down in the object named `data_augmentation_chain` is just one possibility of what a data augmentation pipeline for unform-size images could look like. Feel free to put together other image processing chains, you can use the `DataAugmentationConstantInputSize` class as a template. Or you could use the original SSD data augmentation pipeline by instantiting an `SSDDataAugmentation` object and passing that to the generator instead. This procedure is not exactly efficient, but it evidently produces good results on multiple datasets.

An `SSDInputEncoder` object, `ssd_input_encoder`, is passed to both the training and validation generators. As explained above, it matches the ground truth labels to the model's anchor boxes and encodes the box coordinates into the format that the model needs.

### Note:

The example setup below was used to train SSD7 on two road traffic datasets released by [Udacity](https://github.com/udacity/self-driving-car/tree/master/annotations) with around 20,000 images in total and 5 object classes (car, truck, pedestrian, bicyclist, traffic light), although the vast majority of the objects are cars. The original datasets have a constant image size of 1200x1920 RGB. I consolidated the two datasets, removed a few bad samples (although there are probably many more), and resized the images to 300x480 RGB, i.e. to one sixteenth of the original image size. In case you'd like to train a model on the same dataset, you can download the consolidated and resized dataset I used [here](https://drive.google.com/open?id=1tfBFavijh4UTG4cGqIKwhcklLXUDuY0D) (about 900 MB).
"""

# 1: Instantiate two `DataGenerator` objects: One for training, one for validation.

# Optional: If you have enough memory, consider loading the images into memory for the reasons explained above.
if args.command=='train':
    train_dataset = DataGenerator(load_images_into_memory=False, hdf5_dataset_path=None)
    val_dataset = DataGenerator(load_images_into_memory=False, hdf5_dataset_path=None)

    # 2: Parse the image and label lists for the training and validation datasets.


    # ours DAS AUTO
    image_dir= '../TACO-master/data/'


    #DAS AUTO
    train_labels_filename='map10_without_batch10_15_val.json'
    val_labels_filename='map10_without_batch10_15_val.json'

    train_dataset.parse_json(images_dirs=[image_dir],
                             annotations_filenames=[train_labels_filename],
                             ground_truth_available=True,
                             include_classes='all',
                             ret=False,
                             verbose=True
                             )
    val_dataset.parse_json(images_dirs=[image_dir],
                             annotations_filenames=[val_labels_filename],
                             ground_truth_available=True,
                             include_classes='all',
                             ret=False,
                             verbose=True
                             )


    # Get the number of samples in the training and validations datasets.
    train_dataset_size = train_dataset.get_dataset_size()
    val_dataset_size   = val_dataset.get_dataset_size()

    print("Number of images in the training dataset:\t{:>6}".format(train_dataset_size))
    print("Number of images in the validation dataset:\t{:>6}".format(val_dataset_size))


    # 3: Set the batch size.

    batch_size = config["batch_size"]

    # 4: Define the image processing chain.

    data_augmentation_chain = DataAugmentationConstantInputSize(random_brightness=(-48, 48, 0.5),
                                                                random_contrast=(0.5, 1.8, 0.5),
                                                                random_saturation=(0.5, 1.8, 0.5),
                                                                random_hue=(18, 0.5),
                                                                random_flip=0.5,
                                                                random_translate=((0.03,0.5), (0.03,0.5), 0.5),
                                                                random_scale=(0.5, 2.0, 0.5),
                                                                n_trials_max=3,
                                                                clip_boxes=True,
                                                                overlap_criterion='area',
                                                                bounds_box_filter=(0.3, 1.0),
                                                                bounds_validator=(0.5, 1.0),
                                                                n_boxes_min=1,
                                                                background=(0,0,0))

    # 5: Instantiate an encoder that can encode ground truth labels into the format needed by the SSD loss function.

    # The encoder constructor needs the spatial dimensions of the model's predictor layers to create the anchor boxes.
    predictor_sizes = [model.get_layer('classes11').output_shape[1:3],
                       model.get_layer('classes4').output_shape[1:3],
                       model.get_layer('classes5').output_shape[1:3],
                       model.get_layer('classes6').output_shape[1:3],
                       model.get_layer('classes7').output_shape[1:3]]

    ssd_input_encoder = SSDInputEncoder(img_height=img_height,
                                        img_width=img_width,
                                        n_classes=n_classes,
                                        predictor_sizes=predictor_sizes,
                                        scales=scales,
                                        aspect_ratios_global=aspect_ratios,
                                        two_boxes_for_ar1=two_boxes_for_ar1,
                                        steps=steps,
                                        offsets=offsets,
                                        clip_boxes=clip_boxes,
                                        variances=variances,
                                        matching_type='multi',
                                        pos_iou_threshold=0.5,
                                        neg_iou_limit=0.3,
                                        normalize_coords=normalize_coords)

    # 6: Create the generator handles that will be passed to Keras' `fit_generator()` function.

    #DAS-AUTO : adding for resize here from to image_height and image_width
    resize=Resize(width=img_width,height=img_height)


    train_generator = train_dataset.generate(batch_size=batch_size,
                                             shuffle=True,
                                             transformations=[resize,data_augmentation_chain],
                                             label_encoder=ssd_input_encoder,
                                             returns={'processed_images',
                                                      'encoded_labels'},
                                             keep_images_without_gt=False)

    val_generator = val_dataset.generate(batch_size=batch_size,
                                         shuffle=False,
                                         transformations=[resize],
                                         label_encoder=ssd_input_encoder,
                                         returns={'processed_images',
                                                  'encoded_labels'},
                                         keep_images_without_gt=False)

    # print(keras.__version__)

    """## 4. Set the remaining training parameters and train the model

    We've already chosen an optimizer and a learning rate and set the batch size above, now let's set the remaining training parameters.

    I'll set a few Keras callbacks below, one for early stopping, one to reduce the learning rate if the training stagnates, one to save the best models during the training, and one to continuously stream the training history to a CSV file after every epoch. Logging to a CSV file makes sense, because if we didn't do that, in case the training terminates with an exception at some point or if the kernel of this Jupyter notebook dies for some reason or anything like that happens, we would lose the entire history for the trained epochs. Feel free to add more callbacks if you want TensorBoard summaries or whatever.
    """

    # Define model callbacks.

# TODO: Set the filepath under which you want to save the weights.
    model_checkpoint = ModelCheckpoint(filepath='ssd7_v2_epoch-{epoch:02d}_loss-{loss:.4f}_val_loss-{val_loss:.4f}.h5',
                                       monitor='val_loss',
                                       verbose=1,
                                       save_best_only=True,
                                       save_weights_only=False,
                                       mode='auto',
                                       period=1)

    csv_logger = CSVLogger(filename='ssd7_training_log.csv',
                           separator=',',
                           append=True)

    early_stopping = EarlyStopping(monitor='val_loss',
                                   min_delta=0.0,
                                   patience=10,
                                   verbose=1)

    reduce_learning_rate = ReduceLROnPlateau(monitor='val_loss',
                                             factor=0.2,
                                             patience=8,
                                             verbose=1,
                                             epsilon=0.001,
                                             cooldown=0,
                                             min_lr=0.00001)

    callbacks = [model_checkpoint,
                 csv_logger,
                 early_stopping,
                 reduce_learning_rate]

    """I'll set one epoch to consist of 1,000 training steps I'll arbitrarily set the number of epochs to 20 here. This does not imply that 20,000 training steps is the right number. Depending on the model, the dataset, the learning rate, etc. you might have to train much longer to achieve convergence, or maybe less.

    Instead of trying to train a model to convergence in one go, you might want to train only for a few epochs at a time.

    In order to only run a partial training and resume smoothly later on, there are a few things you should note:
    1. Always load the full model if you can, rather than building a new model and loading previously saved weights into it. Optimizers like SGD or Adam keep running averages of past gradient moments internally. If you always save and load full models when resuming a training, then the state of the optimizer is maintained and the training picks up exactly where it left off. If you build a new model and load weights into it, the optimizer is being initialized from scratch, which, especially in the case of Adam, leads to small but unnecessary setbacks every time you resume the training with previously saved weights.
    2. You should tell `fit_generator()` which epoch to start from, otherwise it will start with epoch 0 every time you resume the training. Set `initial_epoch` to be the next epoch of your training. Note that this parameter is zero-based, i.e. the first epoch is epoch 0. If you had trained for 10 epochs previously and now you'd want to resume the training from there, you'd set `initial_epoch = 10` (since epoch 10 is the eleventh epoch). Furthermore, set `final_epoch` to the last epoch you want to run. To stick with the previous example, if you had trained for 10 epochs previously and now you'd want to train for another 10 epochs, you'd set `initial_epoch = 10` and `final_epoch = 20`.
    3. Callbacks like `ModelCheckpoint` or `ReduceLROnPlateau` are stateful, so you might want ot save their state somehow if you want to pick up a training exactly where you left off.
    """

    # TODO: Set the epochs to train for.
    # If you're resuming a previous training, set `initial_epoch` and `final_epoch` accordingly.


    initial_epoch   = 0
    final_epoch     = 1
    steps_per_epoch = 2

    history = model.fit_generator(generator=train_generator,
                                  steps_per_epoch=steps_per_epoch,
                                  epochs=final_epoch,
                                  callbacks=callbacks,
                                  validation_data=val_generator,
                                  validation_steps=ceil(val_dataset_size/batch_size),
                                  initial_epoch=initial_epoch)

    """
    Let's look at how the training and validation loss evolved to check whether our training is going in the right direction:
    """

    plt.figure(figsize=(20,12))
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')

    plt.legend(loc='upper right', prop={'size': 24});
    plt.savefig('graph_history.png')

"""
### 5. Make predictions
"""

# 1: Set the generator for the predictions.
if args.command=="evaluate":

    orig_images = [] # Store the images here.
    input_images = [] # Store resized versions of the images here.

    # We'll only load one image in this example.
    img_path = args.evaluate_img # default is one image in images/img.jpg

    orig_images.append(imread(img_path))
    img = image.load_img(img_path, target_size=(img_height, img_width))
    img = image.img_to_array(img)
    input_images.append(img)
    input_images = np.array(input_images)

    # model prediction
    y_pred = model.predict(input_images)


    y_pred_decoded = decode_detections(y_pred,
                                   confidence_thresh=0.1,
                                   iou_threshold=0.45,
                                   top_k=10,
                                   normalize_coords=normalize_coords,
                                   img_height=img_height,
                                   img_width=img_width)
    # this can be changed
    confidence_threshold = 0.0

    #y_pred_thresh = [y_pred[k][y_pred[k,:,1] > confidence_threshold] for k in range(y_pred_decoded.shape[0])]

    np.set_printoptions(precision=2, suppress=True, linewidth=90)
    print("Predicted boxes:\n")
    print('   class   conf xmin   ymin   xmax   ymax')
    print(y_pred_decoded[0])

    # Display the image and draw the predicted boxes onto it.

    # Set the colors for the bounding boxes
    colors = plt.cm.hsv(np.linspace(0, 1, 21)).tolist()

    plt.figure(figsize=(20,12))
    plt.imshow(orig_images[0])

    current_axis = plt.gca()

    for box in y_pred_decoded[0]:
        # Transform the predicted bounding boxes for the 300x300 image to the original image dimensions.
        xmin = box[2] * orig_images[0].shape[1] / img_width
        ymin = box[3] * orig_images[0].shape[0] / img_height
        xmax = box[4] * orig_images[0].shape[1] / img_width
        ymax = box[5] * orig_images[0].shape[0] / img_height
        color = colors[int(box[0])]
        label = '{}: {:.2f}'.format(classes[int(box[0])], box[1])
        current_axis.add_patch(plt.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, color=color, fill=False, linewidth=2))
        current_axis.text(xmin, ymin, label, size='x-large', color='white', bbox={'facecolor':color, 'alpha':1.0})

    img_save=list(img_path.split('/'))
    saved_output="output/output_{}".format(img_save[-1])
    print("Saving......{}".format(saved_output))
    plt.savefig(saved_output)
