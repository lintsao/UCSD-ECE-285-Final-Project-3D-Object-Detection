# ECE-285 Final Project: Complex YOLOv4
Welcome to the ECE-285 Final Project repository. This guide will walk you through the steps to preprocess the data, train the model, evaluate the performance, and visualize the results using Complex YOLO - YOLO V4. The PyTorch Implementation based on YOLOv4 of the paper: [Complex-YOLO: Real-time 3D Object Detection on Point Clouds](https://arxiv.org/pdf/1803.06199.pdf)

[![Python](https://img.shields.io/badge/python-v3.6%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/pytorch-v1.5.0%2B-red)](https://pytorch.org/)

---
## Features
- [x] Realtime 3D object detection based on YOLOv4
- [x] Support [distributed data parallel training](https://github.com/pytorch/examples/tree/master/distributed/ddp)
- [x] Tensorboard
- [x] Mosaic/Cutout augmentation for training
- [x] Use [GIoU](https://arxiv.org/pdf/1902.09630v2.pdf) loss of rotated boxes for optimization.

## Complex-YOLO architecture
This work has been based on the paper [YOLOv4: Optimal Speed and Accuracy of Object Detection](https://arxiv.org/abs/2004.10934).
Please refer to several implementations of YOLOv4 using PyTorch DL framework:
- [Tianxiaomo/pytorch-YOLOv4](https://github.com/Tianxiaomo/pytorch-YOLOv4)
- [Ultralytics/yolov3_and_v4](https://github.com/ultralytics/yolov3)
- [WongKinYiu/PyTorch_YOLOv4](https://github.com/WongKinYiu/PyTorch_YOLOv4)
- [VCasecnikovs/Yet-Another-YOLOv4-Pytorch](https://github.com/VCasecnikovs/Yet-Another-YOLOv4-Pytorch)

## Usage
### Requirement
```sh
pip install -U -r requirements.txt
```
For [`mayavi`](https://docs.enthought.com/mayavi/mayavi/installation.html) and [`shapely`](https://shapely.readthedocs.io/en/latest/project.html#installing-shapely) 
libraries, please refer to the installation instructions from their official websites.

### Preprocessing
- **Pretrained Model**
   - Download the pretrained model from [Download Model](https://drive.google.com/file/d/1pyTkEUCtPrLWGHkDaviZNlM7U13eCmkg/view?usp=sharing)
- **Setup the model**
   - Create a `checkpoints` folder and move the downloaded pretrained model to the `checkpoints` folder.
  ```sh
  mkdir checkpoints
  ```
  
### Dataset
- **Download the datasets**:
   - Download the 3D KITTI detection dataset from [here](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d). The downloaded data includes:
      - Velodyne point clouds _**(29 GB)**_: input data to the Complex-YOLO model
      - Training labels of object data set _**(5 MB)**_: input label to the Complex-YOLO model
      - Camera calibration matrices of object data set _**(16 MB)**_: for visualization of predictions
      - Left color images of object data set _**(12 GB)**_: for visualization of predictions

   - For 3D point cloud preprocessing, please refer to the previous works:
      - [VoxelNet-Pytorch](https://github.com/skyhehe123/VoxelNet-pytorch)
      - [Complex-YOLOv2](https://github.com/AI-liu/Complex-YOLO)
      - [Complex-YOLOv3](https://github.com/ghimiredhikura/Complex-YOLOv3)
   - Or You can directly download the prepared datasets from
      - [Dataset Part 1](https://drive.google.com/file/d/17NrrnO-Uw_foiGlPtXXr3IalQUhmZxGz/view?usp=sharing)
      - [Dataset Part 2](https://drive.google.com/file/d/1_WIldJYOrxmVVSjm0no10Llg8vi3VstE/view?usp=sharing)
- **Unzip and organize the datasets**:
   - Unzip the downloaded datasets.
   - Move the unzipped datasets to the current root folder.

- **Please make sure that you construct the source code & dataset directories structure as below**
```
${ROOT}
└── checkpoints/    
    ├── complex_yolov3/
    └── complex_yolov4/
└── dataset/    
    └── kitti/
        ├──ImageSets/
        │   ├── train.txt
        │   └── val.txt
        ├── training/
        │   ├── image_2/ <-- for visualization
        │   ├── calib/
        │   ├── label_2/
        │   └── velodyne/
        └── testing/  
        │   ├── image_2/ <-- for visualization
        │   ├── calib/
        │   └── velodyne/ 
        └── classes_names.txt
└── src/
    ├── config/
    ├── cfg/
        │   ├── complex_yolov3.cfg
        │   ├── complex_yolov3_tiny.cfg
        │   ├── complex_yolov4.cfg
        │   ├── complex_yolov4_tiny.cfg
    │   ├── train_config.py
    │   └── kitti_config.py
    ├── data_process/
    │   ├── kitti_bev_utils.py
    │   ├── kitti_dataloader.py
    │   ├── kitti_dataset.py
    │   ├── kitti_data_utils.py
    │   ├── train_val_split.py
    │   └── transformation.py
    ├── models/
    │   ├── darknet2pytorch.py
    │   ├── darknet_utils.py
    │   ├── model_utils.py
    │   ├── yolo_layer.py
    └── utils/
    │   ├── evaluation_utils.py
    │   ├── iou_utils.py
    │   ├── logger.py
    │   ├── misc.py
    │   ├── torch_utils.py
    │   ├── train_utils.py
    │   └── visualization_utils.py
    ├── evaluate.py
    ├── test.py
    ├── test.sh
    ├── train.py
    └── train.sh
├── README.md 
└── requirements.txt
```

### Training from the pretrained weights
To train the Complex YOLO - YOLO V4 model from scratch or the pretrained weights, use the following command:

```sh
python train.py --model_def config/cfg/complex_yolov4.cfg --pretrained_path checkpoints/complex_yolo_yolo_v4.pth --save_path checkpoints/complex_yolo_yolo_v4.pth
```

### Evaluation
To evaluate the model and calculate the mean Average Precision (mAP), run:

```sh
python eval_mAP.py --model_def config/cfg/complex_yolov4.cfg --pretrained_path checkpoints/complex_yolo_yolo_v4.pth
```

### Visualization
To visualize the detection results, use the following command:

```sh
python detection.py --model_def config/cfg/complex_yolov4.cfg --pretrained_path checkpoints/complex_yolo_yolo_v4.pth
```

## Reference
- https://github.com/RichardMinsooGo-ML/Bible_3_32_Pytorch_Complex_Yolo_Yolov4/tree/main
- https://github.com/maudzung/Complex-YOLOv4-Pytorch
- https://wikidocs.net/181734















