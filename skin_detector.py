import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from tensorflow.keras import backend as K
from keras.utils import np_utils
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.models import model_from_json
import sys

def load_one_image(path):
    im = ImageOps.grayscale(Image.open(fp=path))
    return im

def resize_one_image(im, target_size):
    '''
        target_size : tuple (height, width)
    '''
    im = im.resize(size=target_size)
    im = np.array(im)
    return im[:, :, np.newaxis]

def rescale_one_image(im):
    im = im.astype('float32')
    return im/255

def expand_one_image(im):
    '''
        model.predict() expects a batch, not a single point. Resize to (batch, height, width, channel)
    '''
    return np.expand_dims(im, axis=0)
    
def load_model_architecture(path='/home/ajaquani/Skin_detector/models/skin2.json'):
    with open(path, 'r') as filin:
        json_string = [line for line in filin]
    model = model_from_json(json_string=json_string[0])
    return model

if __name__ == '__main__' :
    print(f'Executing script {sys.argv[0]}')	
    if len(sys.argv) < 2:
        sys.exit('Error : you have to specify an image path as argument')
    
    im = load_one_image(path=sys.argv[1])
    im = resize_one_image(im, target_size=(224, 224))
    im = rescale_one_image(im)
    im = expand_one_image(im)
    
    model = load_model_architecture()
    model.load_weights(filepath='/home/ajaquani/Skin_detector/models/skin2.h5')
    model.compile(optimizer=SGD(learning_rate=0.01), loss='binary_crossentropy', metrics=['accuracy'])
    
    print(f'\n\nP(malignant) = {model.predict(im)[0][1]*100:.2f}%')
