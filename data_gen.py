import json

import cv2 as cv
from torch.utils.data import Dataset
from torchvision import transforms

# Data augmentation and normalization for training
# Just normalization for validation
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.125, contrast=0.125, saturation=0.125),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
    'valid': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}


class ImgClsDataset(Dataset):
    def __init__(self, split):
        filename = '{}.json'.format(split)
        with open(filename, 'r') as file:
            samples = json.load(file)

        self.samples = samples

        self.transformer = data_transforms[split]

    def __getitem__(self, i):
        sample = self.samples[i]
        full_path = sample['img_path']
        label = sample['label']
        img = cv.imread(full_path)

        img = img[..., ::-1]  # RGB
        img = transforms.ToPILImage()(img)
        img = self.transformer(img)
        return img, label

    def __len__(self):
        return len(self.samples)


if __name__ == "__main__":
    train = ImgClsDataset('train')
    print('num_train: ' + str(len(train)))
    valid = ImgClsDataset('valid')
    print('num_valid: ' + str(len(valid)))

    print(train[0])
    print(valid[0])
