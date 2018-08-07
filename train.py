import os
import glob
import struct
import argparse
from shutil import rmtree
from struct import unpack
import pickle

import tensorflow as tf
import numpy as np
from numpy.random import RandomState
import cv2
from PIL import Image
from PIL import ImageDraw
from tqdm import tqdm

# import matplotlib.pyplot as plt
# from sklearn.preprocessing import LabelEncoder
# from keras.layers import BatchNormalization, Conv2D, MaxPool2D, Flatten, Dense, Dropout
# from keras.utils.io_utils import HDF5Matrix
# from keras.utils.np_utils import to_categorical

from keras.models import Sequential
from keras import applications, optimizers
from keras.layers import Input
from keras.models import Model
from keras.metrics import top_k_categorical_accuracy
from keras.utils.training_utils import multi_gpu_model
from keras.callbacks import *

class CyclicLR(Callback):
    """This callback implements a cyclical learning rate policy (CLR).
    The method cycles the learning rate between two boundaries with
    some constant frequency, as detailed in this paper (https://arxiv.org/abs/1506.01186).
    The amplitude of the cycle can be scaled on a per-iteration or 
    per-cycle basis.
    This class has three built-in policies, as put forth in the paper.
    "triangular":
        A basic triangular cycle w/ no amplitude scaling.
    "triangular2":
        A basic triangular cycle that scales initial amplitude by half each cycle.
    "exp_range":
        A cycle that scales initial amplitude by gamma**(cycle iterations) at each 
        cycle iteration.
    For more detail, please see paper.
    
    # Example
        ```python
            clr = CyclicLR(base_lr=0.001, max_lr=0.006,
                                step_size=2000., mode='triangular')
            model.fit(X_train, Y_train, callbacks=[clr])
        ```
    
    Class also supports custom scaling functions:
        ```python
            clr_fn = lambda x: 0.5*(1+np.sin(x*np.pi/2.))
            clr = CyclicLR(base_lr=0.001, max_lr=0.006,
                                step_size=2000., scale_fn=clr_fn,
                                scale_mode='cycle')
            model.fit(X_train, Y_train, callbacks=[clr])
        ```    
    # Arguments
        base_lr: initial learning rate which is the
            lower boundary in the cycle.
        max_lr: upper boundary in the cycle. Functionally,
            it defines the cycle amplitude (max_lr - base_lr).
            The lr at any cycle is the sum of base_lr
            and some scaling of the amplitude; therefore 
            max_lr may not actually be reached depending on
            scaling function.
        step_size: number of training iterations per
            half cycle. Authors suggest setting step_size
            2-8 x training iterations in epoch.
        mode: one of {triangular, triangular2, exp_range}.
            Default 'triangular'.
            Values correspond to policies detailed above.
            If scale_fn is not None, this argument is ignored.
        gamma: constant in 'exp_range' scaling function:
            gamma**(cycle iterations)
        scale_fn: Custom scaling policy defined by a single
            argument lambda function, where 
            0 <= scale_fn(x) <= 1 for all x >= 0.
            mode paramater is ignored 
        scale_mode: {'cycle', 'iterations'}.
            Defines whether scale_fn is evaluated on 
            cycle number or cycle iterations (training
            iterations since start of cycle). Default is 'cycle'.
    """

    def __init__(self, base_lr=0.001, max_lr=0.006, step_size=2000., mode='triangular',
                 gamma=1., scale_fn=None, scale_mode='cycle'):
        super(CyclicLR, self).__init__()

        self.base_lr = base_lr
        self.max_lr = max_lr
        self.step_size = step_size
        self.mode = mode
        self.gamma = gamma
        if scale_fn == None:
            if self.mode == 'triangular':
                self.scale_fn = lambda x: 1.
                self.scale_mode = 'cycle'
            elif self.mode == 'triangular2':
                self.scale_fn = lambda x: 1/(2.**(x-1))
                self.scale_mode = 'cycle'
            elif self.mode == 'exp_range':
                self.scale_fn = lambda x: gamma**(x)
                self.scale_mode = 'iterations'
        else:
            self.scale_fn = scale_fn
            self.scale_mode = scale_mode
        self.clr_iterations = 0.
        self.trn_iterations = 0.
        self.history = {}

        self._reset()

    def _reset(self, new_base_lr=None, new_max_lr=None,
               new_step_size=None):
        """Resets cycle iterations.
        Optional boundary/step size adjustment.
        """
        if new_base_lr != None:
            self.base_lr = new_base_lr
        if new_max_lr != None:
            self.max_lr = new_max_lr
        if new_step_size != None:
            self.step_size = new_step_size
        self.clr_iterations = 0.
        
    def clr(self):
        cycle = np.floor(1+self.clr_iterations/(2*self.step_size))
        x = np.abs(self.clr_iterations/self.step_size - 2*cycle + 1)
        if self.scale_mode == 'cycle':
            return self.base_lr + (self.max_lr-self.base_lr)*np.maximum(0, (1-x))*self.scale_fn(cycle)
        else:
            return self.base_lr + (self.max_lr-self.base_lr)*np.maximum(0, (1-x))*self.scale_fn(self.clr_iterations)
        
    def on_train_begin(self, logs={}):
        logs = logs or {}

        if self.clr_iterations == 0:
            K.set_value(self.model.optimizer.lr, self.base_lr)
        else:
            K.set_value(self.model.optimizer.lr, self.clr())        
            
    def on_batch_end(self, epoch, logs=None):
        
        logs = logs or {}
        self.trn_iterations += 1
        self.clr_iterations += 1

        self.history.setdefault('lr', []).append(K.get_value(self.model.optimizer.lr))
        self.history.setdefault('iterations', []).append(self.trn_iterations)

        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)
        
        K.set_value(self.model.optimizer.lr, self.clr())

def top_5_accuracy(x,y): 
    return top_k_categorical_accuracy(x,y, 5)

def unpack_drawing(file_handle):
    key_id, = unpack('Q', file_handle.read(8))
    countrycode, = unpack('2s', file_handle.read(2))
    recognized, = unpack('b', file_handle.read(1))
    timestamp, = unpack('I', file_handle.read(4))
    n_strokes, = unpack('H', file_handle.read(2))
    image = []
    for i in range(n_strokes):
        n_points, = unpack('H', file_handle.read(2))
        fmt = str(n_points) + 'B'
        x = unpack(fmt, file_handle.read(n_points))
        y = unpack(fmt, file_handle.read(n_points))
        image.append((x, y))

    return {
        'key_id': key_id,
        'countrycode': countrycode,
        'recognized': recognized,
        'timestamp': timestamp,
        'image': image
    }

def dist(a, b):
    return np.power((np.power((a[0] - b[0]), 2) + np.power((a[1] - b[1]), 2)), 1./2)

def min_max(coords):
    x, y = [], []
    for i in range(len(coords)):
        x.append(int(min(coords[i][0]))); x.append(int(max(coords[i][0])))
        y.append(int(min(coords[i][1]))); y.append(int(max(coords[i][1])))
    return min(x), max(x), min(y), max(y)

def norm(image):
    return image.astype('float32') / 255.

class QDPrep:

    def __init__(self, path, to_drop, random_state=42, chunksize=64, max_dataset_size=5000000, trsh=100, normed=True,
                    train_portion=0.9, k=0.05, min_points=3, min_edges=3, dotSize=3, offset=5, img_size=(64,64)):
        self.prng = RandomState(random_state)
        self.dotSize = dotSize
        self.offset = offset + dotSize//2
        self.trsh = trsh
        self.normed = normed
        self.img_size = img_size
        self.max_dataset_size = max_dataset_size
        self.train_portion = int(max_dataset_size * train_portion)
        self.min_edges = min_edges
        self.min_points = min_points
        self.path = path
        self.k = k
        self.chunksize = chunksize
        self.classes = [f.split('/')[-1].split('.')[0] for f in glob.glob(os.path.join(self.path, '*.bin'))]
        self.classes = {k:i for i, k in enumerate(self.classes) if k not in to_drop}
        self.imgs_per_class = max_dataset_size // len(self.classes)
        with open(self.path + '/classes.json', 'w') as f:
            json.dump(self.classes, f)
        self.names = []
        self.binaries = {}
        for key in tqdm(self.classes, desc='read classes binaries', ascii=True):
            self.binaries[key] = [i['image'] for i in list(self.unpack_drawings('%s/%s.bin' % (self.path, key)))]
            self.names.extend([key+'_'+str(i) for i in range(len(self.binaries[key]))])
        self.prng.shuffle(self.names)
        print(" [INFO] %s files & %s classes prepared" % (len(self.names), len(self.classes)))

    def unpack_drawings(self, filename):
        with open(filename, 'rb') as f:
            i = 0
            while i <= self.imgs_per_class:
                i += 1
                try:
                    yield unpack_drawing(f)
                except struct.error:
                    break

    def OHE(self, y):
        if type(y) != int:
            ohe = np.zeros((len(y), len(self.classes)))
            ohe[np.arange(len(y)), y.astype('int64')] = 1
        else:
            ohe = np.zeros(len(self.classes))
            ohe[y] = 1
        return ohe

    def edges_counter(self, coords):
        try:
            coords = np.array(coords).astype(np.int32)
        except:
            coords = np.concatenate(coords, axis=1)  
        shape = coords.shape
        if coords.ndim > 2:
            coords = np.concatenate(coords, axis=1)
        if shape[-1] > shape[-2]:
            coords = coords.T
        elif coords.ndim == 2 and type(coords[0][0]) == tuple:
            coords = list(coords)
            for i in range(shape[0]):
                coords[i] = np.array([coords[i][0], coords[i][1]])
            coords = np.concatenate(coords, axis=1).T
        try:
            p = cv2.arcLength(coords, closed=False)
            points = cv2.approxPolyDP(coords, epsilon=p*self.k, closed=False)
            npoints = len(points) - 1
        except:
            npoints = 0
        return npoints

    def quickdraw_coords2img(self, image):
        image = np.array([[list(j) for j in i] for i in image])
        if self.img_size:
            min_dists, dists = {}, [[] for i in range(len(image))]
            for i in range(len(image)):
                for j in range(len(image[i][0])):
                    dists[i].append(dist([0, 0], [image[i][0][j], image[i][1][j]]))
                min_dists[min(dists[i])] = i

            min_dist = min(list(min_dists.keys()))
            min_index = min_dists[min_dist]
            start_point = [image[min_index][0][dists[min_index].index(min_dist)], image[min_index][1][dists[min_index].index(min_dist)]]
            for i in range(len(image)):
                for j in range(len(image[i][0])):
                    image[i][0][j] = image[i][0][j] - start_point[0]
                    image[i][1][j] = image[i][1][j] - start_point[1]

            min_x, max_x, min_y, max_y = min_max(image) 
            scaleX = ((max_x - min_x) / (self.img_size[0]-(self.offset*2-1)))
            scaleY = ((max_y - min_y) / (self.img_size[1]-(self.offset*2-1)))
            for i in range(len(image)):
                for j in range(len(image[i][0])):
                    image[i][0][j] = image[i][0][j] / scaleX
                    image[i][1][j] = image[i][1][j] / scaleY

        min_x, max_x, min_y, max_y = min_max(image)
        img = Image.new("RGB", (max_x-min_x+self.offset*2, max_y-min_y+self.offset*2), "white")
        draw = ImageDraw.Draw(img)

        for j in range(len(image)):
            for i in range(len(image[j][0]))[1:]:
                x, y = image[j][0][i-1], image[j][1][i-1]
                x_n, y_n = image[j][0][i], image[j][1][i]
                x -= min_x-self.offset; y -= min_y-self.offset
                x_n -= min_x-self.offset; y_n -= min_y-self.offset
                draw.line([(x,y), (x_n,y_n)], fill="black", width=self.dotSize)

        if self.img_size:
            return {'img':img, 'scaleX':scaleX, 'scaleY':scaleY, 'start_point': start_point}
        return {'img':img}

    def run_generator(self, val_mode=False):
        pics, targets, i, n = [], [], 0, 0
        lims = [0, self.train_portion]
        if val_mode:
            lims = [self.train_portion, None]
        length = len(self.names[lims[0]:lims[1]])
        N = length // self.chunksize
        while True:
            for name in self.names[lims[0]:lims[1]]:
                class_name, no = name.split('_')
                target = self.classes[class_name]
                coords = self.binaries[class_name][int(no)]

                if sum([[len(k) for k in j][0] for j in coords]) < self.min_points:
                    continue
                if self.min_edges:
                    if self.edges_counter(coords) < self.min_edges:
                        continue

                img = np.array(self.quickdraw_coords2img(coords)['img'])
                img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                img = cv2.bitwise_not(img)
                img = cv2.resize(img, self.img_size, Image.LANCZOS)
                img = cv2.threshold(img, self.trsh, 255,
                        cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

                if self.normed:
                    img = norm(img)

                img = img[:,:,np.newaxis]
                pics.append(img)
                targets.append(self.OHE(target))
                i += 1
                if n == N and i == (length % self.chunksize):
                    yield (np.array(pics), np.array(targets))
                        
                elif i == self.chunksize:
                    out_pics, out_target = np.array(pics), np.array(targets)
                    pics, targets, i = [], [], 0
                    n += 1
                    yield (out_pics, out_target)

from tensorflow.python.client import device_lib
def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']

if __name__ == '__main__':

    #python3 train.py --G 1 -p /home/shapes/first_level/quickdraw -n /home/quick-draw_classifier/models

    parser = argparse.ArgumentParser(description='quickDraw classifier')
    parser.add_argument('-g', '--G', type=int, default=1)
    parser.add_argument('-p', '--path')
    parser.add_argument('-n', '--name')
    args = parser.parse_args()
    globals().update(vars(args))

    print("[INFO] GPU devices:%s" % get_available_gpus())

    try:
        rmtree(name)
    except:
        pass
    os.mkdir(name)

    ################################################################################

    batch_size = 64 * G
    nbepochs = 3
    img_size = (64,64)
    reader = QDPrep(path, [], random_state=42, chunksize=batch_size, 
                              max_dataset_size=1000000, trsh=100, normed=True,
                              train_portion=0.9, k=0.05, min_points=10, 
                              min_edges=3, dotSize=3, offset=5, img_size=img_size)

    ################################################################################

    nclasses = len(reader.classes)
    if G <= 1:
        print("[INFO] training with 1 GPU...")
        model = applications.mobilenetv2.MobileNetV2(
            include_top=True, classes=nclasses, weights=None, input_tensor=Input(shape=img_size+(1,)))
    else:
        print("[INFO] training with {} GPUs...".format(G))
     
        with tf.device("/cpu:0"):
            model = applications.mobilenetv2.MobileNetV2(
                include_top=True, classes=nclasses, weights=None, input_tensor=Input(shape=img_size+(1,)))
        model = multi_gpu_model(model, gpus=G)

    adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-7, decay=0.0, clipnorm=5)
    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=["accuracy", top_5_accuracy])
    model.summary()

    with open(name + '/model_summary.txt','w') as f:
            model.summary(print_fn=lambda x: f.write(x + '\n'))
    model_json = model.to_json()
    with open(name+"/model.json", "w") as json_file:
        json_file.write(model_json)

    train_steps = reader.train_portion // batch_size
    val_steps = (reader.max_dataset_size - reader.train_portion) // batch_size

    checkpoint = ModelCheckpoint(name+'/checkpoint_weights.h5', monitor='val_loss', verbose=1, 
                     save_best_only=True, mode='min', save_weights_only=True)
    clr = CyclicLR(base_lr=0.001, max_lr=0.006, step_size=train_steps*2, mode='exp_range', gamma=0.99994)

    print("[INFO] training network...")

    H = model.fit_generator(reader.run_generator(val_mode=False),
            steps_per_epoch=train_steps, epochs=nbepochs, shuffle=False, verbose=1,
            validation_data=reader.run_generator(val_mode=True), validation_steps=val_steps,
            use_multiprocessing=False, workers=1, callbacks=[checkpoint, clr])

    pickle.dump(H.history, open(name+'/loss_history.pickle.dat', 'wb'))
    print("[INFO] Finished!")