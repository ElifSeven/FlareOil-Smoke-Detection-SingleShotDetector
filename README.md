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
       
6. Check remaining GPU time

7. Create Label Map (.pbtxt)
-       item {
         id: 1
         name: 'Flare oil'
        }

       item {
         id: 2
         name: 'Smoke'
       }
8. Convert XML to CSV file(.csv)
- With the images labeled, we need to create TFRecords that can be served as input data for training the object detector.
- To create the TFRecords, we will first convert the XML label files created with LabelImg to one CSV file using the [xml_to_csv.py ](https://github.com/ElifSeven/SSD-final/blob/main/xml_to_csv.py) script.
-      !python xml_to_csv.py

- The above command creates two files in the images directory. 
- One is called test_labels.csv, and another one is called train_labels.csv. Next, I will convert the CSV files into TFRecords files.

9. Create TFRecord (.record)
- I use [generate_tfrecord.py : ](https://github.com/ElifSeven/SSD-final/blob/main/generate_tfrecord.py) to convert our data set into train.record and test.record.
-     !python generate_tfrecord.py --csv_input=images/train_labels.csv --image_dir=images/train --output_path=train.record
      !python generate_tfrecord.py --csv_input=images/test_labels.csv --image_dir=images/test --output_path=test.record

10. Modify Config (.config) File
- Each of the pretrained models has a config file that contains details about the model. To detect our custom class, the config file needs to be modified accordingly.The config files are included in the models directory you cloned in the very beginning.
-  You can find them in:
-      models/research/object_detection/samples/configs
In our case, I will modify the config file for **ssd_mobilenet_v1_pets.config**. Make a copy of it first and save it in the models/ directory.
Here are the items you need to change:
1. **num_classes** to 2
2. **fine_tune_checkpoin**t tells the model which checkpoint file to use. Set this to **ssd_mobilenet_v1_coco_2018_01_28/model.ckpt**
3. The model also need to know where the TFRecord files and label maps are both training and test sets.
-     train_input_reader: {
      tf_record_input_reader {
      input_path: "/content/gdrive/MyDrive/models/research/object_detection/data/train.record"
      }

      }

      eval_config: {
      num_examples: 450
      }

      eval_input_reader: {
      tf_record_input_reader {
      input_path: "/content/gdrive/MyDrive/models/research/object_detection/data/test.record"
      }
      label_map_path: "/content/gdrive/MyDrive/models/research/object_detection/training/object-detection.pbtxt"
      shuffle: false
      num_readers: 1
       }

11. Train
-      !pip install tf_slim
       %cd /content/gdrive/MyDrive/models/research/object_detection
       os.environ['PYTHONPATH'] += ':/content/gdrive/MyDrive/models/research/:/content/gdrive/MyDrive/models/research/slim'
       !python train.py --train_dir=training/ --pipeline_config_path=training/ssd_mobilenet_v1_pets.config --logtostderr
 12. Model export
 - Once you finish training your model, you can export your model to be used for inference
 -     - !python export_inference_graph.py 
          --input_type image_tensor 
          --pipeline_config_path training/ssd_mobilenet_v1_pets.config 
          --trained_checkpoint_prefix  training/model.ckpt-6325 
          --output_directory new_graph
 13. Test
 - You need to move [custom_model_images.py](https://github.com/ElifSeven/SSD-final/blob/main/custom_model_images.py) and
 [custom_model_video.py](https://github.com/ElifSeven/SSD-final/blob/main/custom_model_video.py) to /models/research/object_detection/ path. Then you have to run:
 -     !python /content/gdrive/MyDrive/models/research/object_detection/custom_model_images.py

