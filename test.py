import os
import time

import cv2 as cv
import keras.backend as K
import numpy as np
from tqdm import tqdm

from utils import load_model

if __name__ == '__main__':
    model = load_model()

    print('Predicting test data')
    num_samples = 8041
    start = time.time()
    out = open('result.txt', 'a')
    for i in tqdm(range(num_samples)):
        filename = os.path.join('data/test', '%05d.jpg' % (i + 1))
        bgr_img = cv.imread(filename)
        rgb_img = cv.cvtColor(bgr_img, cv.COLOR_BGR2RGB)
        rgb_img = np.expand_dims(rgb_img, 0)
        preds = model.predict(rgb_img)
        prob = np.max(preds)
        class_id = np.argmax(preds)
        out.write('{}\n'.format(str(class_id + 1)))

    end = time.time()
    seconds = end - start
    print('avg fps: {}'.format(str(num_samples / seconds)))

    out.close()
    K.clear_session()
