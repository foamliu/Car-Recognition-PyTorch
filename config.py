import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # sets device for model and PyTorch tensors

im_size = 224
num_classes = 196
num_channels = 3

train_data = 'data/train'
valid_data = 'data/test'

num_train_samples = 6549
num_valid_samples = 1595
verbose = 1
batch_size = 32
num_epochs = 1000
patience = 50

# Training parameters
num_workers = 4  # for data-loading
grad_clip = 5.  # clip gradients at an absolute value of
print_freq = 100  # print training/validation stats  every __ batches
checkpoint = None  # path to checkpoint, None if none
