import torch
from torchvision import datasets, transforms
from my_nn_lib_torch import Conv2d, LeckyReLU, Linear, Softmax, Flatten, MyModel, CrossEntropyLoss


transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)



model = MyModel([Conv2d(3, 16, (7, 7), 1, 0), 
                LeckyReLU(),
                Conv2d(16, 32, (5, 5), 1, 0),
                LeckyReLU(),
                Conv2d(32, 8, (3, 3), 1, 0),
                LeckyReLU(),
                Flatten(),
                Linear(8*20*20, 10),
                Softmax()])

hyper_params = {    
    'lr': 0.005,
    'epoch': 50,
    'batch_size': 32,
    'alpha': 0.9
}

model.train_with_dataset(train_dataset, CrossEntropyLoss, hyper_params, show_plot=True)

