# Car Recognition


This repository is to do car recognition by fine-tuning ResNet-152 with Cars Dataset from Stanford.


## Dependencies

- Python 3.6.8
- PyTorch 1.3

## Dataset

We use the Cars Dataset, which contains 16,185 images of 196 classes of cars. The data is split into 8,144 training images and 8,041 testing images, where each class has been split roughly in a 50-50 split.

 ![image](https://github.com/foamliu/Car-Recognition-PyTorch/raw/master/images/random.jpg)

You can get it from [Cars Dataset](https://ai.stanford.edu/~jkrause/cars/car_dataset.html):

```bash
$ cd Car-Recognition-PyTorch
$ wget http://imagenet.stanford.edu/internal/car196/cars_train.tgz
$ wget http://imagenet.stanford.edu/internal/car196/cars_test.tgz
$ wget --no-check-certificate https://ai.stanford.edu/~jkrause/cars/car_devkit.tgz
```

## ImageNet Pretrained Models

Download [ResNet-152](https://drive.google.com/file/d/0Byy2AcGyEVxfeXExMzNNOHpEODg/view?usp=sharing) into models folder.

## Usage

### Data Pre-processing
Extract 8,144 training images, and split them by 80:20 rule (6,515 for training, 1,629 for validation):
```bash
$ python pre_process.py
```

### Train
```bash
$ python train.py
```

If you want to visualize during training, run in your terminal:
```bash
$ tensorboard --logdir runs
```

 ![image](https://github.com/foamliu/Car-Recognition-PyTorch/raw/master/images/train.jpg)

### Analysis
Update "model_weights_path" in "utils.py" with your best model, and use 1,629 validation images for result analysis:
```bash
$ python analyze.py
```

#### Validation acc:
**88.70%**

#### Confusion matrix:

 ![image](https://github.com/foamliu/Car-Recognition-PyTorch/raw/master/images/confusion_matrix.jpg)

### Test
```bash
$ python test.py
```

Submit predictions of test data set (8,041 testing images) at [Cars Dataset](https://ai.stanford.edu/~jkrause/cars/car_dataset.html), evaluation result:

#### Test acc:
**88.88%**

 ![image](https://github.com/foamliu/Car-Recognition-PyTorch/raw/master/images/test.jpg)

### Demo

![image](https://github.com/foamliu/Car-Recognition-PyTorch/raw/master/images/samples/07647.jpg)

```bash
class_name: Lamborghini Reventon Coupe 2008
prob: 0.9999994
```

Download [pre-trained model](https://github.com/foamliu/Car-Recognition-PyTorch/releases/download/v1.0/car_recognition.pt) then run:

```bash
$ python demo.py
```
It picks 20 random valid images to recognize:


1 | 2 | 3 | 4 |
|---|---|---|---|
|![image](https://github.com/foamliu/Car-Recognition-PyTorch/raw/master/images/0_out.png)  | ![image](https://github.com/foamliu/Car-Recognition-PyTorch/raw/master/images/1_out.png) | ![image](https://github.com/foamliu/Car-Recognition-PyTorch/raw/master/images/2_out.png)|![image](https://github.com/foamliu/Car-Recognition-PyTorch/raw/master/images/3_out.png) |
|$(result_0)|$(result_1)|$(result_2)|$(result_3)|
|![image](https://github.com/foamliu/Car-Recognition-PyTorch/raw/master/images/4_out.png)  | ![image](https://github.com/foamliu/Car-Recognition-PyTorch/raw/master/images/5_out.png) | ![image](https://github.com/foamliu/Car-Recognition-PyTorch/raw/master/images/6_out.png)|![image](https://github.com/foamliu/Car-Recognition-PyTorch/raw/master/images/7_out.png) |
|$(result_4)|$(result_5)|$(result_6)|$(result_7)|
|![image](https://github.com/foamliu/Car-Recognition-PyTorch/raw/master/images/8_out.png)  | ![image](https://github.com/foamliu/Car-Recognition-PyTorch/raw/master/images/9_out.png) | ![image](https://github.com/foamliu/Car-Recognition-PyTorch/raw/master/images/10_out.png)|![image](https://github.com/foamliu/Car-Recognition-PyTorch/raw/master/images/11_out.png)|
|$(result_8)|$(result_9)|$(result_10)|$(result_11)|
|![image](https://github.com/foamliu/Car-Recognition-PyTorch/raw/master/images/12_out.png) | ![image](https://github.com/foamliu/Car-Recognition-PyTorch/raw/master/images/13_out.png)| ![image](https://github.com/foamliu/Car-Recognition-PyTorch/raw/master/images/14_out.png)|![image](https://github.com/foamliu/Car-Recognition-PyTorch/raw/master/images/15_out.png)|
|$(result_12)|$(result_13)|$(result_14)|$(result_15)|
|![image](https://github.com/foamliu/Car-Recognition-PyTorch/raw/master/images/16_out.png) | ![image](https://github.com/foamliu/Car-Recognition-PyTorch/raw/master/images/17_out.png)|![image](https://github.com/foamliu/Car-Recognition-PyTorch/raw/master/images/18_out.png) | ![image](https://github.com/foamliu/Car-Recognition-PyTorch/raw/master/images/19_out.png)|
|$(result_16)|$(result_17)|$(result_18)|$(result_19)|
