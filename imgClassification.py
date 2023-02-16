# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 16:07:18 2021

@author: yangxing
"""
from datetime import datetime
import glob
import os
#os.environ['CUDA_VISIBLE_DEVICES']='1'
import tensorflow as tf
import numpy as np
import random

batch_size = 8  # 256 is best for RTX 3090 (per GPU)
total_epoch = 150
class_num = 16
img_lists = []
labels = []

img_folder = '' # user define, like 'D:/data/vidClip/S06-classified-sample' 
folder_list = os.listdir(img_folder)
for i,folder in enumerate(folder_list):
    img_list = glob.glob(os.path.join(img_folder,folder,'*.jpg'))
    for img_path in img_list:
        img_lists.append(img_path)
        label = np.zeros(class_num)
        ind = i%class_num
        label[ind] = 1
        labels.append(label)

randnum = random.randint(0,100)
random.seed(randnum)
random.shuffle(img_lists)
random.seed(randnum)
random.shuffle(labels)
        
count = 0  
count_test = 0     
train_num = int(0.8*len(labels))
test_num = len(labels)-train_num
index = list(np.arange(len(labels)))
def generate_arrays_from_image(img_lists,labels):
    global count, batch_size, index
    while 1:
        data = []
        label = []
        for i in range(batch_size):
            ind = count%train_num#len(labels)
            if ind==0:
                random.shuffle(index)
            ind = index[ind]
            img = tf.io.read_file(img_lists[ind])
            img = tf.image.decode_image(img)
            img = tf.image.resize(img, [576, 1024])
            data.append(img)
            label.append(labels[ind])
            count = count + 1
        data = np.array(data, dtype=np.float32)
        label = np.array(label)
        data = tf.convert_to_tensor(data)                                                 
        label = tf.convert_to_tensor(label)
        yield(data, label)    


def generate_test_arrays_from_image(img_lists,labels):
    global count_test, batch_size, index
    while 1:
        data = []
        label = []
        for i in range(batch_size):
            ind = count_test%test_num#len(labels)
            ind = index[train_num+ind]
            img = tf.io.read_file(img_lists[ind])
            img = tf.image.decode_image(img)
            img = tf.image.resize(img, [576, 1024])
            data.append(img)
            label.append(labels[ind])
            count_test = count_test + 1
        data = np.array(data, dtype=np.float32)
        label = np.array(label)
        data = tf.convert_to_tensor(data)                                                 
        label = tf.convert_to_tensor(label)
        yield(data, label)  
        
model = tf.keras.applications.ResNet50(weights=None,input_shape=(576,1024,3),classes=class_num)

time_pre_train = '' # user define, like'20210702-184658'#'20210408-114204'
best_pre_train = 101
checkpoint_path = "checkpoints/{time:s}/cp-{epoch:03d}.ckpt"
model.load_weights(checkpoint_path.format(time=time_pre_train, epoch=best_pre_train))
 
 
loss_func = tf.keras.losses.BinaryCrossentropy()

optimizer_func=tf.keras.optimizers.Adam(learning_rate=0.001)
time_now = datetime.now().strftime("%Y%m%d-%H%M%S")
if not os.path.exists("./checkpoints/"+time_now):
    os.system("mkdir .\\checkpoints\\"+time_now)

checkpoint_path = "checkpoints/last/cp-{epoch:03d}.ckpt"
cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,save_weights_only=True,verbose=1,save_freq=len(labels))

logs = "logs/"+time_now
tboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logs,histogram_freq=1)

# print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), end="  Start train.\n")
print(datetime.now().strftime("%Y%m%d-%H%M%S"), end="  Start train.\n")

model.compile(optimizer=optimizer_func,loss=loss_func,metrics=['accuracy'] )
# if train_mode:
#     model.fit(train_dataset,epochs=total_epoch-continue_train,validation_data=test_dataset,callbacks=[cp_callback, tboard_callback])
# else:
#     model.fit(train_dataset,epochs=total_epoch-continue_train,validation_data=test_dataset,callbacks=[cp_callback, tboard_callback])

model.fit_generator(generate_arrays_from_image(img_lists,labels),steps_per_epoch=round(train_num/batch_size),
                    epochs=total_epoch, validation_data=generate_test_arrays_from_image(img_lists,labels), 
                    validation_steps=round(test_num/batch_size), shuffle=True, callbacks=[cp_callback, tboard_callback])
                    # workers=2,use_multiprocessing=True)

model.summary()

saved_model_path = "./saved_models/{}".format(time_now)
# model.save(saved_model_path)
tf.keras.models.save_model(model, saved_model_path,save_format="tf")
print("Finish training.")
os.system("move .\\checkpoints\\last\\* .\\checkpoints\\"+time_now)