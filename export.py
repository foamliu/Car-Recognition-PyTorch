import time

import torch

from models import CarRecognitionModel

if __name__ == '__main__':
    checkpoint = 'BEST_checkpoint.tar'
    print('loading {}...'.format(checkpoint))
    start = time.time()
    checkpoint = torch.load(checkpoint)
    print('elapsed {} sec'.format(time.time() - start))
    model = checkpoint['model'].module

    filename = 'car_recognition.pt'
    print('saving {}...'.format(filename))
    start = time.time()
    torch.save(model.state_dict(), filename)
    print('elapsed {} sec'.format(time.time() - start))

    print('loading {}...'.format(filename))
    start = time.time()
    model = CarRecognitionModel()
    model.load_state_dict(torch.load(filename))
    print('elapsed {} sec'.format(time.time() - start))

    # filename_scripted = 'gaze_estimation_scripted.pt'
    # print('saving {}...'.format(filename_scripted))
    # start = time.time()
    # torch.jit.save(torch.jit.script(model), filename_scripted)
    # print('elapsed {} sec'.format(time.time() - start))
    #
    # print('loading {}...'.format(filename))
    # start = time.time()
    # model = torch.jit.load(filename_scripted)
    # print('elapsed {} sec'.format(time.time() - start))
