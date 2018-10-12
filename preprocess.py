import random
import dlib, cv2, os
import pandas as pd
import numpy as np

img_size = 224
dirname = 'CAT_06'
base_path = '/mnt/sda1/cats/%s' % dirname
file_list = sorted(os.listdir(base_path))
random.shuffle(file_list)

dataset = {
  'imgs': [],
  'lmks': [],
  'bbs': []
}

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
  if '.cat' not in f:
    continue

  # read landmarks
  pd_frame = pd.read_csv(os.path.join(base_path, f), sep=' ', header=None)
  landmarks = (pd_frame.as_matrix()[0][1:-1]).reshape((-1, 2))

  # load image
  img_filename, ext = os.path.splitext(f)

  img = cv2.imread(os.path.join(base_path, img_filename))

  # resize image and relocate landmarks
  img, ratio, top, left = resize_img(img)
  landmarks = ((landmarks * ratio) + np.array([left, top])).astype(np.int)
  bb = np.array([np.min(landmarks, axis=0), np.max(landmarks, axis=0)])

  dataset['imgs'].append(img)
  dataset['lmks'].append(landmarks.flatten())
  dataset['bbs'].append(bb.flatten())

  # for l in landmarks:
  #   cv2.circle(img, center=tuple(l), radius=1, color=(255, 255, 255), thickness=2)

  # cv2.imshow('img', img)
  # if cv2.waitKey(0) == ord('q'):
  #   break

np.save('dataset/%s.npy' % dirname, np.array(dataset))