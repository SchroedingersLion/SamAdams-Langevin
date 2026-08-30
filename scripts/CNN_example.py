import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import torch.nn as nn
import numpy as np
from Samplers import BAOAB, ZBAOABZ


#%% MODEL.
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)   # Output: (32, 28, 28)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # Halves the size
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)  # Output: (64, 14, 14)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1) # Output: (128, 7, 7)
        
        # Calculate the size after convolutions and pooling
        self.fc_input_size = 128 * 3 * 3  # For input into the first fully connected layer

        # Fully connected layers
        self.fc1 = nn.Linear(self.fc_input_size, 512)  # 128 channels, 3x3 feature map
        self.fc2 = nn.Linear(512, 256)                 # Hidden layer
        self.fc3 = nn.Linear(256, 10)                  # Output layer for 10 classes

    def forward(self, x):
        x = self.pool(nn.ReLU()(self.conv1(x)))  # Output: (batch_size, 32, 14, 14)
        x = self.pool(nn.ReLU()(self.conv2(x)))  # Output: (batch_size, 64, 7, 7)
        x = self.pool(nn.ReLU()(self.conv3(x)))  # Output: (batch_size, 128, 3, 3)

        # Flatten the tensor for fully connected layers
        x = x.view(x.size(0), -1)   # Flatten to (batch_size, 128 * 3 * 3)
        x = nn.ReLU()(self.fc1(x))  # Output: (batch_size, 512)
        x = nn.ReLU()(self.fc2(x))  # Output: (batch_size, 256)
        x = self.fc3(x)             # Output: (batch_size, 10)
        
        x = torch.log_softmax(x, dim=1)
        return x


#%% SETTINGS.
num_classes = 10
method = "ZBAOABZ" # "BAOAB" or "ZBAOABZ"
B = 10000          # large batch size
epochs = 10
meas_freq = 1
lr = 2e-3
lr_schedule = None
gamma = 1
T = 1
weight_decay = 1e-5

alpha = 500
omega = 60_000
dtau = 2e-3
m = 0.1
M = 10

cuda_idx = "0"      # Main GPU index to use for training. Adjust based on your system.
dev_ids = [0]       # list of GPU indices to use for DataParallel. Adjust based on your system.


torch.manual_seed(1)
num_workers = 5

criterion = torch.nn.NLLLoss(reduction="mean")


#%% LOAD DATA.
transfos = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.MNIST(".", train = True, download = True, transform = transfos)
test_dataset = datasets.MNIST(".", train = False, download = True, transform = transfos)


torch.cuda.empty_cache()
device = torch.device('cuda:'+str(cuda_idx) if torch.cuda.is_available() else 'cpu')


#% TRAIN MODEL - SINGLE RUN.
model = SimpleCNN()

if device.type == 'cuda':
    println("Using GPU for training.")
    model = nn.DataParallel(model, device_ids = dev_ids)  ## wrap in DataParallel for multiple GPUs.

model.to(device)

# Create data loaders
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=B, shuffle=True, num_workers=num_workers, pin_memory=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=num_workers, pin_memory=True)


if method == "BAOAB":
    sampler = BAOAB(model, train_loader, test_loader, criterion, lr, weight_decay, gamma, T, epochs, device, meas_freq, lr_schedule=lr_schedule)
    (loss_train, accu_train, accu_test) = sampler.train()
    print("Train accu / test accu:", accu_train, accu_test)
elif method == "ZBAOABZ":
    sampler = ZBAOABZ(model, train_loader, test_loader, criterion, dtau, weight_decay, gamma, alpha, omega, m, M, T, epochs, device, meas_freq, lr_schedule=lr_schedule)
    (loss_train, accu_train, accu_test, dt, zetas) = sampler.train()
    print("Train accu / test accu:", accu_train, accu_test)
else:
    raise ValueError("Not a valid training method!")


#%% PLOT RESULTS.
x_axis_epochs = np.arange(1, len(loss_train) + 1) * meas_freq
# Create a separate x-axis for the batch-level measurements (zetas and dt)
x_axis_batches = np.linspace(0, epochs, len(zetas))

fig, ax = plt.subplots(2, 2)

ax[0][0].plot(x_axis_epochs, loss_train)
ax[0][0].set_title("Loss")

ax[1][0].plot(x_axis_epochs, accu_test)
ax[1][0].set_title("Test Accuracy")

if method == "ZBAOABZ":
    ax[0][1].plot(x_axis_batches, zetas)
    ax[0][1].set_title(r"$\zeta$")

    ax[1][1].plot(x_axis_batches, dt)
    ax[1][1].set_title(r"$\Delta t$")

for a in ax.flat:
    a.set_xlabel("Epochs")

plt.tight_layout()
plt.show()