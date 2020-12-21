# -*- coding: utf-8 -*-
# Use the commands to make use of the code in the repository

# python3 run.py train --dataset=../TACO/data 
# python3 run.py train --dataset=../TACO/data --model=model_weights/Model_v2.h5

# python3 run.py train --dataset=../TACO/data --annot_train=map10_without_batch10_15_train.json --annot_val=map10_without_batch10_15_val.json
# python3 run.py train --dataset=../TACO/data --config=config_v2.json --model=model_weights/Model_v2.h5 --annot_train=custom_map_train_lskw_rm10_15.json --annot_val=custom_map_val_lskw.json


# python3 run.py evaluate --evaluate_img=../TACO/data/batch_14/000001.jpg --model=model_weights/Model.h5 --config=config.json
# python3 run.py evaluate --evaluate_img=../TACO/data/batch_14/000001.jpg --model=model_weights/Model_v2.h5 --config=config_v2.json


# making necessary imports
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
    # parser.add_argument('--pretrained_weights', required=False,default=False,type=bool, help="True or False")
    parser.add_argument('--config', required=False,default="config.json",metavar="/path/config.json", help="Path to config json")    
    parser.add_argument('--model', required=False,default="model_weights/Model.h5",metavar="/path/model.h5", help="Path to weights .h5 file")
    parser.add_argument('--annot_train', required=False,default="None",metavar="/path/to/train_annot.json", help="Path to train annotation file")
    parser.add_argument('--annot_val', required=False,default="None",metavar="/path/to/val_annot.json", help="Path to val samples annotation file")
    parser.add_argument('--dataset', required=False, metavar="/path/dir", help='Directory of the dataset',default='../TACO-master/data')
    parser.add_argument('--lrate', required=False, default=0.001, type=float, help='learning rate')
    parser.add_argument('--evaluate_img',required=False,default='images/img.jpg',metavar="/path/dir", help='Directory of the dataset')
   
    # parser.add_argument('--class_map', required=True, metavar="/path/file.csv", help=' Target classes')
   
    DEFAULT_LOGS_DIR="/training_logs"
    args = parser.parse_args()

    print("-------------------------")
    print("Command: ", args.command)
    # print("pretrained_weights: ",args.pretrained_weights)
    if (not args.annot_train=="None") and (not args.annot_val=="None"):
        print("annot_train: ",args.annot_train)
        print("annot_train: ",args.annot_val)
    print("config: ", args.config)
    print("Model: ", args.model)
    print("Dataset: ", args.dataset)
    print("Logs: ", DEFAULT_LOGS_DIR)
    print("-------------------------")


## 1. Set the model configuration parameters in the config.json file

# with open('config.json') as f:
with open(args.config) as f:    
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
# currently not supported
# if not args.pretrained_weights:
#     model.load_weights(args.model, by_name=True)

# 3: Instantiate an Adam optimizer and the SSD loss function and compile the model

adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

ssd_loss = SSDLoss(neg_pos_ratio=3, alpha=1.0)

model.compile(optimizer=adam, loss=ssd_loss.compute_loss)



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

# 3. Set up the data generators for the training


#Instantiate two `DataGenerator` objects: One for training, one for validation.

if args.command=='train':
    train_dataset = DataGenerator(load_images_into_memory=False, hdf5_dataset_path=None)
    val_dataset = DataGenerator(load_images_into_memory=False, hdf5_dataset_path=None)

    # 2: Parse the image and label lists for the training and validation datasets.


    # ours DAS AUTO
    image_dir= '../TACO/data/'


    #DAS AUTO
    # train_labels_filename='map10_without_batch10_15_val.json'
    # val_labels_filename='map10_without_batch10_15_val.json'
    if args.annot_train=="None" or args.annot_val=="None":
        # print("Error annotation file not found for either val or train")
        raise NameError("Error annotation file not found for either val or train") 
        
    train_labels_filename=args.annot_train
    val_labels_filename=args.annot_val


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


    # Set the batch size.

    batch_size = config["batch_size"]

    # Define the image processing chain.

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

    #Instantiate an encoder that can encode ground truth labels into the format needed by the SSD loss function.

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

    #Create the generator handles that will be passed to Keras' `fit_generator()` function.

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



    # 4. Set the remaining training parameters and train the model


    # Define model callbacks.


    model_checkpoint = ModelCheckpoint(filepath='model_weights/sd7_v2_epoch-{epoch:02d}_loss-{loss:.4f}_val_loss-{val_loss:.4f}.h5',
                                       monitor='val_loss',
                                       verbose=1,
                                       save_best_only=True,
                                       save_weights_only=False,
                                       mode='auto',
                                       period=1)

    csv_logger = CSVLogger(filename='training_logs/sd7_training_log.csv',
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

    #observing the loss
    plt.figure(figsize=(20,12))
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')

    plt.legend(loc='upper right', prop={'size': 24})
    plt.savefig('graph_history.png')

# Set the generator for the evaluation on some images
if args.command=="evaluate":

    orig_images = [] # Store the images here.
    input_images = [] # Store resized versions of the images here.

    # image path as argument to the python file
    img_path = args.evaluate_img # default is one image in images/img.jpg

    orig_images.append(imread(img_path))
    img = image.load_img(img_path, target_size=(img_height, img_width))
    img = image.img_to_array(img)
    input_images.append(img)
    input_images = np.array(input_images)

    # model prediction
    y_pred = model.predict(input_images)


    y_pred_decoded = decode_detections(y_pred,
                                   confidence_thresh=0.08,
                                   iou_threshold=0.45,
                                   top_k=10,
                                   normalize_coords=normalize_coords,
                                   img_height=img_height,
                                   img_width=img_width)
    # this can be changed
    # confidence_threshold = 0.0
    print(y_pred)

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

    # saving the image to the output folder
    img_save=list(img_path.split('/'))
    model_name=(args.model.split('/')[-1]).split('.')[0]
    saved_output="output/{}_output_{}".format(model_name,img_save[-1])
    print("Saving......{}".format(saved_output))
    plt.savefig(saved_output)
