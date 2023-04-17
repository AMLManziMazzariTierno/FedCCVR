import torch
from torchvision import models
from torch import nn
import torch.nn.functional as F
import torch.nn.init as init

NUM_GROUP = 8
# GroupNorm takes number of groups to divide the channels in and the number of channels to expect
# in the input. 

class MLP(nn.Module):

    def __init__(self, n_feature, n_hidden, n_output, dropout=0.5):
        super(MLP, self).__init__()
        self.dropout = torch.nn.Dropout(dropout)

        self.hidden_1 = torch.nn.Linear(n_feature, n_hidden)  # hidden layer
        self.bn1 = torch.nn.BatchNorm1d(n_hidden)

        self.hidden_2 = torch.nn.Linear(n_hidden, n_hidden // 2)
        self.bn2 = torch.nn.BatchNorm1d(n_hidden // 2)

        self.hidden_3 = torch.nn.Linear(n_hidden // 2, n_hidden // 4)  # hidden layer
        self.bn3 = torch.nn.BatchNorm1d(n_hidden // 4)

        self.hidden_4 = torch.nn.Linear(n_hidden // 4, n_hidden // 8)  # hidden layer
        self.bn4 = torch.nn.BatchNorm1d(n_hidden // 8)

        self.out = torch.nn.Linear(n_hidden // 8, n_output)  # output layer

    def forward(self, X):
        x = X.view(X.shape[0], -1)

        x = F.relu(self.hidden_1(x))  # hidden layer 1
        x = self.dropout(self.bn1(x))

        x = F.relu(self.hidden_2(x))  # hidden layer 2
        x = self.dropout(self.bn2(x))

        x = F.relu(self.hidden_3(x))  # hidden layer 3
        x = self.dropout(self.bn3(x))

        x = F.relu(self.hidden_4(x))  # hidden layer 4
        feature = self.dropout(self.bn4(x))

        x = self.out(feature)

        return x, feature


class CNN_Model(nn.Module):

    def __init__(self):
        super(CNN_Model, self).__init__()
        self.cnn = nn.Sequential(
            ## Layer 1
            nn.Conv2d(3, 6, 5, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            ## Layer 2
            nn.Conv2d(6, 16, 5, 1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.fc = nn.Sequential(

            ##Layer 3
            nn.Linear(400,120),
            nn.ReLU(),

            #Layer 4
            nn.Linear(120,84),
            nn.ReLU(),

            #Layer 5
            nn.Linear(84,84),
            nn.ReLU(),

            #Layer 6
            nn.Linear(84,256)
        )

        #Layer 7  classifier layer
        self.classifier = nn.Linear(256,10)

    def forward(self, input):
        x = self.cnn(input)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x, self.classifier(x)



def _weights_init(m):
    classname = m.__class__.__name__
    #print(classname)
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
    
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.gn1 = nn.GroupNorm(NUM_GROUP, planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.gn2 = nn.GroupNorm(NUM_GROUP, planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                  nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                  nn.GroupNorm(NUM_GROUP, self.expansion * planes)  )
          
        self.relu = nn.ReLU()
        self.stride = stride

    def forward(self, x):
        out = self.conv1(x)
        out = self.gn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.gn2(out)
        out += self.shortcut(x)
        out = self.relu(out)
        return out

class Resnet20(nn.Module):
    """implementation of ResNet20 with GN layers"""
    def __init__(self, lr, device, n_classes=100, input_shape = (28,28)):
    #def __init__(self, num_classes=100):
      super(Resnet20, self).__init__()
      block = BasicBlock
      num_blocks = [3,3,3]
      self.num_classes = n_classes
      self.device = device
      self.lr = lr
      self.in_planes = 16
      self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
      self.gn1 = nn.GroupNorm(NUM_GROUP, 16)
      self.relu = nn.ReLU()
      
      self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
      self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
      self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
      self.linear = nn.Linear(64, n_classes)

      self.apply(_weights_init)
      #self.weights = self.apply(_weights_init)
      self.size = self.model_size()
      print(f"size definito {self.size}")

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.gn1(out)
        out = self.relu(out)
        
        out = self.layer1(out)
        out = self.layer2(out)
        feature = self.layer3(out)

        out = torch.nn.functional.avg_pool2d(feature, feature.size()[3])
        out = out.view(out.size(0), -1)
        try:
            out = self.linear(out)
        except:
            out = out
            
        return out, feature
      
    def model_size(self):
        tot_size = 0
        for param in self.parameters():
            tot_size += param.size()[0]
        return tot_size
        
    def summary(self):
        return "summary"

def resnet20(image_channels=3):
    return Resnet20(BasicBlock, [3, 3, 3], image_channels)

class ReTrainModel(nn.Module):

    def __init__(self):
        super(ReTrainModel, self).__init__()

        #Layer 7  classifier layer
        self.classifier = nn.Linear(256,100)

    def forward(self, input):

        return self.classifier(input)

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal(m.weight, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        torch.nn.init.normal(m.weight, 1.0, 0.02)
        torch.nn.init.constant(m.bias, 0.0)
    elif classname.find('Linear') != -1:
        torch.nn.init.normal(m.weight, 0.0, 0.01)
        torch.nn.init.constant(m.bias, 0.0)




