from torch import nn
from torchscope import scope
from torchvision import models

from config import num_classes


class CarRecognitionModel(nn.Module):
    def __init__(self):
        super(CarRecognitionModel, self).__init__()
        resnet = models.resnet50(pretrained=True)
        # Remove linear layer
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        # self.pool = nn.AvgPool2d(kernel_size=7)
        self.fc = nn.Linear(2048, num_classes)
        self.softmax = nn.Softmax()

    def forward(self, images):
        x = self.resnet(images)  # [N, 2048, 1, 1]
        # x = self.pool(x)
        x = x.view(-1, 2048)  # [N, 2048]
        x = self.fc(x)
        x = self.softmax(x)
        return x


if __name__ == "__main__":
    model = CarRecognitionModel()
    scope(model, input_size=(3, 224, 224))
