# SSD-final

#### The repository provides all the files needed to train a "flare oil & smoke" detector that can accurately detect flare oil and smoke. This readme file describes how to replace these files with your own files to train a detection classifier. It also has Python scripts to test your classifier out on an image or video.

## Index of this Repository
- [SingleShotDetector_final : ](https://github.com/ElifSeven/SSD-final/blob/main/SingleShotDetector_final.ipynb) Is the code to train and test tensorflow object detection using a pre-trained SSD MobileNet V2 model.
- [object-detection.pbtxt: ](https://github.com/ElifSeven/SSD-final/blob/main/object-detection.pbtxt) Classes listed in the label map. 
- [xml_to_csv.py : ](https://github.com/ElifSeven/SSD-final/blob/main/xml_to_csv.py) To create a CSV file which contains all the XML files and their bounding box coordinates to single CSV file which is input for creating TFrecords.
- [generate_tfrecord.py : ](https://github.com/ElifSeven/SSD-final/blob/main/generate_tfrecord.py)Before you can train your custom object detector, you must convert your data into the TFRecord format.
- [ssd_mobilenet_v1_pets.config : ](https://github.com/ElifSeven/SSD-final/blob/main/ssd_mobilenet_v1_pets.config)Contains details about the model and contains various arguments.
- [custom_model_images.py : ](https://github.com/ElifSeven/SSD-final/blob/main/custom_model_images.py) To detect image 
- [custom_model_video.py : ](https://github.com/ElifSeven/SSD-final/blob/main/custom_model_video.py) To detection in videos

## Dataset
#### The Dataset for this project was collected. And was labelled manually in xml format.

## Hardware
#### I have used Google Colab to train and test the model.

## HOW TO BEGIN?
- Open my [SingleShotDetector_final ](https://github.com/ElifSeven/SSD-final/blob/main/SingleShotDetector_final.ipynb) on your browser.
- Click on File in the menu bar and click on Save a copy in drive. This will open a copy of my Colab notebook on your browser which you can now use.
- Next, once you have opened the copy of my notebook and are connected to the Google Colab VM, click on Runtime in the menu bar and click on Change runtime type. Select GPU and click on save.

## Instructions
1. Install  dependencies using !pip
-     !pip install --upgrade pip
      !pip install --upgrade protobuf
2. Check GPU status
3. Mount drive, link your folder
4. Clone the Tensorflow models repository
- I use resources in the Tensorflow models repository. Since it does not come with the Tensorflow installation, we need to clone it from their Github repo
-     -git clone https://github.com/tensorflow/models.git
5. Setting up the environment 
- Install protobuf and compile, install setup.py
      !apt-get install protobuf-compiler python-pil python-lxml python-tk
       !pip install Cython
       %cd /content/gdrive/MyDrive/models/research
        protoc object_detection/protos/*.proto --python_out=.

       import os
       os.environ['PYTHONPATH'] += ':/content/gdrive/MyDrive/models/research/:/content/gdrive/MyDrive/models/research/slim'
       !python /content/gdrive/MyDrive/models/research/slim/setup.py build
       !python /content/gdrive/MyDrive/models/research/slim/setup.py install
       
6.Check remaining GPU time
7.Generating Training data
- With the images labeled, we need to create TFRecords that can be served as input data for training the object detector.
- To create the TFRecords, we will first convert the XML label files created with LabelImg to one CSV file using the [xml_to_csv.py ](https://github.com/ElifSeven/SSD-final/blob/main/xml_to_csv.py) script.
-      !python xml_to_csv.py

- The above command creates two files in the images directory. 
- One is called test_labels.csv, and another one is called train_labels.csv. Next, I will convert the CSV files into TFRecords files.


