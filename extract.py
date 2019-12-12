import os
import tarfile


def extract(filename):
    print('Extracting {}...'.format(filename))
    tar = tarfile.open(filename, "r:gz")
    tar.extractall('data')
    tar.close()


if __name__ == '__main__':
    if not os.path.exists('data/cars_train'):
        extract('data/cars_train.tgz')
    if not os.path.exists('data/cars_test'):
        extract('data/cars_test.tgz')
    if not os.path.exists('data/car_devkit'):
        extract('data/car_devkit.tgz')
