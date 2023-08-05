# TS_OAMIL-for-Plant-disease-detection

* This code is an implementation of our manuscript: An Iterative Noisy Annotation Correction Model for Robust Plant Disease Detection.

* The manuscript is currently under peer review. We submitted it to Frontiers in Plant Science.
* Authors: Jiuqing Dong, Alvaro Fuentes, Sook Yoon*, Hyongsuk Kim*, Dong Sun Park
* We will release the code and complete this documentation as soon as possible.


### Installation
* Set up environment

    conda create -n TSOAMIL python=3.7

    conda activate TSOAMIL

    conda install pytorch==1.10.0 torchvision==0.11.0 torchaudio==0.10.0 cudatoolkit=11.3 -c pytorch -c conda-forge

* install dependecies

    pip install -r requirements/build.txt

* install mmcv (will take a while to process)

    cd mmcv
    MMCV_WITH_OPS=1 pip install -e .

    cd ..
    pip install -e .

* Generate noisy annotations: Please refer to files in utils/

  There are two version noise you can choose, random noise and real world noise(default gamma is 0.15).

### Dataset prepare

  You can download the GWHD2021 dataset from this link: https://zenodo.org/record/5092309

The original file annotation file format is '.csv'. You can transfer to a 'xml' file by using this command python ./process_gwhd/excel_to_xml.py

### train

  sh train.sh

### test

  sh test.sh

#### We will optimize our code and complete this documentation as soon as possible.



## How to use this code for a customer dataset?

Operational steps for the teacher-student learning paradigm:

step 1: Inference the training set and save XML files.
(Modify the last few lines of mmcv.visualization.image.py)
(Modify the test path to the training set in configs.base.datasets.paprika_detection_oamil.py)

step 2: Perform XML to JSON conversion.
Run python ./utils/xml_to_json_COCOformat_paprika.py

step 3: Generate the PKL file.
Run python ./utils/gen_update_paprika.py

step 4: Modify the relevant file paths and parameters.
Training path, training parameters, save path, GPU, and port.

Important notes for adding the Control class:
After adding a class, you need to modify the annotation files for train, val, and test sets. Inconsistent classes between training, testing, and validation can cause errors.
When generating the PKL file, modify ./utils/coco_dataset.py to adjust the classes.
The number of classes in mmdet.datasets.samples.coco.py and customer.py also needs to be adjusted.
