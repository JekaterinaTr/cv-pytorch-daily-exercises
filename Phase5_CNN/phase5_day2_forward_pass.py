class CNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 8, 3)
        self.conv2 = nn.Conv2d(8, 16, 3)
        self.relu = nn.ReLU()

        self.fc = nn.Linear(16 * 60 * 60, 10)

    def forward(self, x):

        x = self.conv1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.relu(x)

        x = x.view(x.size(0), -1)

        x = self.fc(x)

        return x


model = CNN()

out = model(torch.rand(1,3,64,64))

print("Model output:", out.shape)