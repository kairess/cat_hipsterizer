import keras, sys, cv2, os
from keras.models import Model, load_model
import numpy as np
import pandas as pd

img_size = 224
base_path = 'samples'
file_list = sorted(os.listdir(base_path))

bbs_model_name = sys.argv[1]
lmks_model_name = sys.argv[2]
bbs_model = load_model(bbs_model_name)
lmks_model = load_model(lmks_model_name)

def resize_img(im):
  old_size = im.shape[:2] # old_size is in (height, width) format
  ratio = float(img_size) / max(old_size)
  new_size = tuple([int(x*ratio) for x in old_size])
  # new_size should be in (width, height) format
  im = cv2.resize(im, (new_size[1], new_size[0]))
  delta_w = img_size - new_size[1]
  delta_h = img_size - new_size[0]
  top, bottom = delta_h // 2, delta_h - (delta_h // 2)
  left, right = delta_w // 2, delta_w - (delta_w // 2)
  new_im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT,
      value=[0, 0, 0])
  return new_im, ratio, top, left

for f in file_list:
  if '.jpg' not in f:
    continue

  img = cv2.imread(os.path.join(base_path, f))
  ori_img = img.copy()

  # predict bounding box
  img, ratio, top, left = resize_img(img)

  inputs = (img.astype('float32') / 255).reshape((1, img_size, img_size, 3))
  pred_bb = bbs_model.predict(inputs)[0].reshape((-1, 2))

  # compute bounding box of original image
  ori_bb = ((pred_bb - np.array([left, top])) / ratio).astype(np.int)

  # compute lazy bounding box for detecting landmarks
  center = np.mean(ori_bb, axis=0)
  face_size = max(np.abs(ori_bb[1] - ori_bb[0]))
  new_bb = np.array([
    center - face_size * 0.6,
    center + face_size * 0.6
  ]).astype(np.int)
  new_bb = np.clip(new_bb, 0, 99999)

  # predict landmarks
  face_img = ori_img[new_bb[0][1]:new_bb[1][1], new_bb[0][0]:new_bb[1][0]]
  face_img, face_ratio, face_top, face_left = resize_img(face_img)

  face_inputs = (face_img.astype('float32') / 255).reshape((1, img_size, img_size, 3))

  pred_lmks = lmks_model.predict(face_inputs)[0].reshape((-1, 2))

  # compute landmark of original image
  new_lmks = ((pred_lmks - np.array([face_left, face_top])) / face_ratio).astype(np.int)
  ori_lmks = new_lmks + new_bb[0]

  # visualize
  cv2.rectangle(ori_img, pt1=tuple(ori_bb[0]), pt2=tuple(ori_bb[1]), color=(255, 255, 255), thickness=2)

  for l in ori_lmks:
    cv2.circle(ori_img, center=tuple(l), radius=1, color=(255, 255, 255), thickness=2)

  cv2.imshow('img', ori_img)

  if cv2.waitKey(0) == ord('q'):
    break