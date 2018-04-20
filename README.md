# Food-calorie-estimation-using-deep-learning
# Installations

## Dependencies

Tensorflow Object Detection API depends on the following libraries:

*   Protobuf 2.6
*   Python-tk
*   Pillow 1.0
*   lxml
*   tf Slim (which is included in the "tensorflow/models/research/" checkout)
*   Jupyter notebook
*   Keras
*   Matplotlib
*   Tensorflow
*   Cython
*   cocoapi
*   ssd

``` bash
# For CPU
pip install tensorflow
# For GPU
pip install tensorflow-gpu
```

## Datasets
 
Dataset used is Fruit-360 [https://www.kaggle.com/moltean/fruits] for image classifications and scrapped images from internet for
object detection

## Protobuf Compilation

The Tensorflow Object Detection API uses Protobufs to configure model and
training parameters. Before the framework can be used, the Protobuf libraries
must be compiled. This should be done by running the following command from
the tensorflow/models/research/ directory:


``` bash
# From tensorflow/models/research/
protoc object_detection/protos/*.proto --python_out=.
```

## Add Libraries to PYTHONPATH

When running locally, the tensorflow/models/research/ and slim directories
should be appended to PYTHONPATH. This can be done by running the following from
tensorflow/models/research/:


``` bash
# From tensorflow/models/research/
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
```

# Testing the Installation

You can test that you have correctly installed the Tensorflow Object Detection\
API by running the following command:

```bash
python object_detection/builders/model_builder_test.py
```

# Configuring the Object Detection Training Pipeline

The Tensorflow Object Detection API uses protobuf files to configure the
training and evaluation process. The schema for the training pipeline can be
found in object_detection/protos/pipeline.proto.

A skeleton configuration file is shown below:

```
model {
(... Add model config here...)
}

train_config : {
(... Add train_config here...)
}

train_input_reader: {
(... Add train_input configuration here...)
}

eval_config: {
}

eval_input_reader: {
(... Add eval_input configuration here...)
}
```

## Defining Inputs

The Tensorflow Object Detection API accepts inputs in the TFRecord file format.
Users must specify the locations of both the training and evaluation files.
Additionally, label map should be specified , which define the mapping
between a class id and class name. The label map should be identical between
training and evaluation datasets.

An example input configuration looks as follows:

```
tf_record_input_reader {
  input_path: "/usr/home/username/data/train.record"
}
label_map_path: "/usr/home/username/data/label_map.pbtxt"
```
# Preparing Inputs

Tensorflow Object Detection API reads data using the TFRecord file format.

## Generating the TFRecord files.
Images are labelled using lblImg [https://github.com/tzutalin/labelImg] which is in xml format. <br/>
XML files are then converted to csv format using sml_to_csv.py [https://github.com/datitran/raccoon_dataset/blob/master/xml_to_csv.py]

Now the folder structure for object detection will look as follows
```bash
Object-Detection
-data/
--test_labels.csv
--train_labels.csv
-images/
--test/
---testingimages.jpg
--train/
---testingimages.jpg
--...yourimages.jpg
-training
-xml_to_csv.py

```

To convert these into TFRecords, run the following commands:

```bash
python generate_tfrecord.py --csv_input=data/train_labels.csv --output_path=data/train.record

python generate_tfrecord.py --csv_input=data/test_labels.csv --output_path=data/test.record
```
# TensorFlow model

model used for object detection is coco trained ssd_mobilenet_v1 which can be downloaded from [https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md]
with the configuration file ssd_mobilenet_v1_pets.config from [https://github.com/tensorflow/models/tree/master/research/object_detection/samples/configs]

start the training with

```bash
python3 train.py --logtostderr --train_dir=training/ --pipeline_config_path=training/ssd_mobilenet_v1_pets.config
```

From models/object_detection, via terminal, start TensorBoard with:
```bash
tensorboard --logdir='training'
```

## Contributors

Chintan Koticha [https://github.com/chintankoticha] <br/>
Chinmay Keskar [https://github.com/keskarCJ] <br/>
Sneha Kawitkar [https://github.com/kawitkars] <br/>

## Copyright

See [LICENSE](LICENSE) for details.
Copyright (c) 2018 
