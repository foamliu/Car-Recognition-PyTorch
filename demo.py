# import the necessary packages
import json
import os
import random

import numpy as np
import scipy.io
import torch
from PIL import Image

from data_gen import data_transforms
from models import CarRecognitionModel

if __name__ == '__main__':
    filename = 'car_recognition.pt'
    model = CarRecognitionModel()
    model.load_state_dict(torch.load(filename))
    model.eval()

    transformer = data_transforms['valid']

    cars_meta = scipy.io.loadmat('data/devkit/cars_meta')
    class_names = cars_meta['class_names']  # shape=(1, 196)
    class_names = np.transpose(class_names)

    test_path = 'data/test/'
    test_images = [f for f in os.listdir(test_path) if
                   os.path.isfile(os.path.join(test_path, f)) and f.endswith('.jpg')]

    num_samples = 20
    samples = random.sample(test_images, num_samples)
    results = []
    for i, image_name in enumerate(samples):
        full_path = os.path.join(test_path, image_name)
        print('Start processing image: {}'.format(full_path))
        img = Image.open(full_path)
        img.save('images/{}_out.png'.format(i))
        img = transformer(img)
        imgs = img.unsqueeze(dim=0)

        with torch.no_grad():
            preds = model(imgs)

        preds = preds.cpu().numpy()[0]
        prob = np.max(preds)
        class_id = np.argmax(preds)
        text = ('Predict: {}, prob: {}'.format(class_names[class_id][0][0], prob))
        results.append({'label': class_names[class_id][0][0], 'prob': '{:.4}'.format(prob)})

    print(results)
    with open('results.json', 'w') as file:
        json.dump(results, file, indent=4)
