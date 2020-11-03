import os
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import shutil
from tensorflow.keras.models import load_model

model = load_model('./model_weights/weight/')

dir = input('Folder with the pics to classify:')
files = os.listdir(dir)
for x in files:
    if x[-4:] == '.jpg':
        img = image.load_img(dir+x,target_size=(28,28))
        img = image.img_to_array(img)
        img = np.expand_dims(img,axis=0)
        img = np.vstack([img])
        classes = model.predict(img)
        img = mpimg.imread(dir+x)
        plt.imshow(img)
        f=max(classes[0])
        print(x,str(list(classes[0]).index(f)))
