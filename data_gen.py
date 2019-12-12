import pickle
from PIL import Image
import cv2 as cv
from torch.utils.data import Dataset
from torchvision import transforms

# Data augmentation and normalization for training
# Just normalization for validation
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomAffine(degrees=20, translate=(0.1, 0.1), scale=(0.8, 1.2)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
    'valid': transforms.Compose([
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}


class CarRecognitionDataset(Dataset):
    def __init__(self, split):
        filename = 'data/{}.pkl'.format(split)
        with open(filename, 'rb') as file:
            samples = pickle.load(file)

        self.samples = samples

        self.transformer = data_transforms[split]

    def __getitem__(self, i):
        sample = self.samples[i]
        full_path = sample['full_path']
        label = sample['label']
        img = Image.open(full_path, 'RGB')
        # img = cv.imread(full_path)
        #
        # img = img[..., ::-1]  # RGB
        # img = transforms.ToPILImage()(img)
        img = self.transformer(img)
        return img, label

    def __len__(self):
        return len(self.samples)


if __name__ == "__main__":
    # train = CarRecognitionDataset('train')
    # print('num_train: ' + str(len(train)))
    # valid = CarRecognitionDataset('valid')
    # print('num_valid: ' + str(len(valid)))
    #
    # print(train[0])
    # print(valid[0])
    filename = 'data/{}.pkl'.format('valid')
    with open(filename, 'rb') as file:
        samples = pickle.load(file)

    transformer = transforms.Compose([
        transforms.RandomAffine(degrees=20, translate=(0.1, 0.1), scale=(0.8, 1.2)),
        transforms.RandomHorizontalFlip(),
    ])

    sample = samples[0]

    full_path = sample['full_path']
    label = sample['label']
    # img = cv.imread(full_path)
    img = Image.open(full_path, 'RGB')
    # img = img[..., ::-1]  # RGB
    # img = transforms.ToPILImage()(img)
    img = transformer(img)

    img.show()

    # cv.imshow('image', img)
    # cv.waitKey(0)
