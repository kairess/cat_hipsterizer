# Cat Hipsterizer
Inspired by ["Hipsterize Your Dog With Deep Learning"](http://blog.dlib.net/2016/10/hipsterize-your-dog-with-deep-learning.html)  

![result.jpg](https://github.com/kairess/cat_hipsterizer/raw/master/result/result.jpg)  
![result2.jpg](https://github.com/kairess/cat_hipsterizer/raw/master/result/result2.jpg)  

Project contains:  
1. Cat face detection with pretrained Mobilenetv2  
2. Cat facial landmarks detection with pretrained Mobilenetv2  
  
At first, I used [cat frontal face detector of OpenCV](https://www.pyimagesearch.com/2016/06/20/detecting-cats-in-images-with-opencv/), but it looks so bad performance for most of real cat photos. So I decided up to make new model with deep learning.  
Regression method is used for both face detection and landmark detection, so that model is very **naive** to use on real application. But it works extremely well than I expected ;)  
  
Used [Cat dataset on Kaggle](https://www.kaggle.com/crawford/cat-dataset) for training and validation.  
  
  
### Cascade Model Structure
1. Input (Full image 224x224) - **Face detection model** - Output (face bounding box)
2. Input (Face image 224x224) - **Facial landmarks model** - Output (9 landmarks points)


# Requirement
- Python
- Keras
- Numpy
- Dlib
- OpenCV
- Pandas

# Usage
### Training

```
python preprocess.py
python train.py
python preprocess_lmks.py
python train_lmks.py
```
### Testing
```
python test.py bbs_1.h5 lmks_1.h5
```

# Limitations
- Detect one cat per frame
- Powerful for frontal faces (a bit low performance for side faces)
- Cannot detect existence, this model thinks cat must be in the picture

# TODOs (for you)
- Multiple cats detection
- Data augmentation (flip, translation, rotation, noise...)
- YOLO like model (class probability map)
- Use transpose convolution layers for landmarks reconstruction (to preserve spatial information)
- Mobile implementation
- Train Dlib shape predictor model
