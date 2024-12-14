import torch
from torchvision import datasets, transforms
from my_nn_lib_torch import Conv2d, LeckyReLU, Linear, Softmax, Flatten, MyModel, CrossEntropyLoss


# 載入訓練和測試資料集
transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)



model = MyModel([Conv2d(3, 9, (7, 7), 1, 0), 
                LeckyReLU(),
                Conv2d(9, 3, (5, 5), 1, 0),
                LeckyReLU(),
                Flatten(),
                Linear(3*22*22, 10),
                Softmax()])

hyper_params = {    
    'lr': 0.005,
    'epoch': 10,
    'batch_size': 64,
    'alpha': 0.9
}
model.train_with_dataset(train_dataset, CrossEntropyLoss, hyper_params, show_plot=True)

