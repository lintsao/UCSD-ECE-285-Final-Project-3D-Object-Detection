# ECE-285-Final

### Usage: 
#### Preprocessing
##### Pretrained Model
- Step 1. Download model from https://drive.google.com/file/d/1pyTkEUCtPrLWGHkDaviZNlM7U13eCmkg/view?usp=sharing
- Step 2. Create "checkpoints" folder and move the pretrained model it

##### Dataset
- Step 1. Download dataset from https://drive.google.com/file/d/17NrrnO-Uw_foiGlPtXXr3IalQUhmZxGz/view?usp=sharing
- Step 2. Download dataset from https://drive.google.com/file/d/1_WIldJYOrxmVVSjm0no10Llg8vi3VstE/view?usp=sharing
- Step 3. Unzip the dataset and move it to the current root folder

#### Training from pretrained weight - Complex Yolo - Yolo V4
```python
python train.py --model_def config/cfg/complex_yolov4.cfg --pretrained_path checkpoints/Complex_yolo_yolo_v4.pth --save_path checkpoints/Complex_yolo_yolo_v4.pth
```

#### Evaluation

```python
python eval_mAP.py --model_def config/cfg/complex_yolov4.cfg --pretrained_path checkpoints/Complex_yolo_yolo_v4.pth
```

#### Visualization

```python
python detection.py --model_def config/cfg/complex_yolov4.cfg --pretrained_path checkpoints/Complex_yolo_yolo_v4.pth
```
