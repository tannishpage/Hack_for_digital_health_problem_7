import torch
import io
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image

# Model Class: 

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels= 3,out_channels= 8, kernel_size= 3 , stride= 1 ,padding = 1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(in_channels = 8, out_channels = 16, kernel_size = 3 , stride =1 , padding = 1 )
        self.conv3 = nn.Conv2d(in_channels = 16, out_channels = 32, kernel_size = 3 , stride =1 , padding = 1 )
        self.conv4 = nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 3 , stride =1 , padding = 1 )
        self.conv5 = nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = 3 , stride =1 , padding = 1 )
        self.conv6 = nn.Conv2d(in_channels = 128, out_channels = 256, kernel_size = 3 , stride =1 , padding = 1 )

        self.fc1 = nn.Linear(256*28*28, 500)
        self.fc2 = nn.Linear(500, 100)
        self.fc3 = nn.Linear(100, 50)
        self.fc4 = nn.Linear(50, 2)


    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = F.relu(self.conv2(x))
        x = self.pool(F.relu(self.conv3(x)))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = self.pool(F.relu(self.conv6(x)))


        x = x.view(-1, 256 * 28 * 28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))

        x = self.fc4(x)
        return x


model = CNN()

device = torch.device('cpu')
model.load_state_dict(torch.load("PATH TO FILE", map_location=device))
model.eval()

# Transform image --> Tensor

def transform_image(image_bytes):
    transform = transformer = transforms.Compose([transforms.Resize((224,224)),
                                  transforms.RandomHorizontalFlip(),
                                  transforms.ToTensor(),
                                  transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
                                ])
    image = Image.open(io.BytesIO(image_bytes))
    return transform(image).unsqueeze(0)

def get_prediction(image_tensor):
    #image = image_tensor.reshape(-1, <Some Value>)
    output = model(image_tensor)
    _, predicted = torch.max(output.data, 1)
    return predicted