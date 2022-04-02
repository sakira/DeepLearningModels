from torch import nn

# define model
class VGG16(nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()
        self.conv11 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1)
        self.conv12 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)

        self.conv21 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.conv22 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)

        self.conv31 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.conv32 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.conv33 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)

        self.conv41 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1)
        self.conv42 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.conv43 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)

        self.conv51 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.conv52 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.conv53 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)

        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(25088, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, 1000)

    def forward(self, x):
        # conv1
        x = self.conv11(x)
        x = self.relu(x)
        x = self.conv12(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # conv2
        x = self.conv21(x)
        x = self.relu(x)
        x = self.conv22(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # conv3
        x = self.conv31(x)
        x = self.relu(x)
        x = self.conv32(x)
        x = self.relu(x)
        x = self.conv33(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # conv4
        x = self.conv41(x)
        x = self.relu(x)
        x = self.conv42(x)
        x = self.relu(x)
        x = self.conv43(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # conv5
        x = self.conv51(x)
        x = self.relu(x)
        x = self.conv52(x)
        x = self.relu(x)
        x = self.conv53(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # Flatten
        x = self.flatten(x)

        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)

        return x


