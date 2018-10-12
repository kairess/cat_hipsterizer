import keras, datetime
from keras.layers import Input, Dense
from keras.models import Model
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau
from keras.applications import mobilenetv2
import numpy as np

img_size = 224

mode = 'lmks' # [bbs, lmks]
if mode is 'bbs':
  output_size = 4
elif mode is 'lmks':
  output_size = 18

start_time = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')

data_01 = np.load('dataset/lmks_CAT_01.npy')
data_02 = np.load('dataset/lmks_CAT_02.npy')
data_03 = np.load('dataset/lmks_CAT_03.npy')
data_04 = np.load('dataset/lmks_CAT_04.npy')
data_05 = np.load('dataset/lmks_CAT_05.npy')
data_06 = np.load('dataset/lmks_CAT_06.npy')

x_train = np.concatenate((data_01.item().get('imgs'), data_02.item().get('imgs'), data_03.item().get('imgs'), data_04.item().get('imgs'), data_05.item().get('imgs')), axis=0)
y_train = np.concatenate((data_01.item().get(mode), data_02.item().get(mode), data_03.item().get(mode), data_04.item().get(mode), data_05.item().get(mode)), axis=0)

x_test = np.array(data_06.item().get('imgs'))
y_test = np.array(data_06.item().get(mode))

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = np.reshape(x_train, (-1, img_size, img_size, 3))
x_test = np.reshape(x_test, (-1, img_size, img_size, 3))

y_train = np.reshape(y_train, (-1, output_size))
y_test = np.reshape(y_test, (-1, output_size))

inputs = Input(shape=(img_size, img_size, 3))

mobilenetv2_model = mobilenetv2.MobileNetV2(input_shape=(img_size, img_size, 3), alpha=1.0, depth_multiplier=1, include_top=False, weights='imagenet', input_tensor=inputs, pooling='max')

net = Dense(128, activation='relu')(mobilenetv2_model.layers[-1].output)
net = Dense(64, activation='relu')(net)
net = Dense(output_size, activation='linear')(net)

model = Model(inputs=inputs, outputs=net)

model.summary()

# training
model.compile(optimizer=keras.optimizers.Adam(), loss='mse')

model.fit(x_train, y_train, epochs=50, batch_size=32, shuffle=True,
  validation_data=(x_test, y_test), verbose=1,
  callbacks=[
    TensorBoard(log_dir='logs/%s' % (start_time)),
    ModelCheckpoint('./models/%s.h5' % (start_time), monitor='val_loss', verbose=1, save_best_only=True, mode='auto'),
    ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, verbose=1, mode='auto')
  ]
)