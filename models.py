import torch.nn as nn
import torch.nn.functional as F
import torch

class FC(nn.Module):
    def __init__(self, hidden):
        super(FC,self).__init__()
        self.net = nn.Sequential(
                nn.Flatten(),
                nn.Linear(784, hidden, bias = False),
                nn.ReLU(),
                # nn.Tanh(),
                nn.Linear(hidden, hidden, bias = False),
                nn.ReLU(),
                # nn.Tanh(),
                nn.Linear(hidden, 10, bias = False),
#               nn.Sigmoid()
                )

    def forward(self,x):
        # output = torch.clamp(self.net(x), max=40) # The maximun value of output is 40
        output = self.net(x)
        return output


class FC_feature(nn.Module):
    def __init__(self, input_zise, H_1, H_2, d):
        super(FC_feature, self).__init__()

        self.feature = [[], [], [], []]
        self.fc1 = nn.Linear(input_zise, H_1, bias = False)
        self.fc2 = nn.Linear(H_1, H_2, bias = False)
        self.fc3 = nn.Linear(H_2, 10, bias = False)
        self.dropout = nn.Dropout(d)

    def forward(self, x):
        layer0_output = x.view(x.shape[0], -1)
        layer1_output = F.relu(self.fc1(layer0_output))
        dropout_output = self.dropout(layer1_output)
        layer2_output = F.relu(self.fc2(dropout_output))
        output = self.fc3(layer2_output)

        self.feature[0] = layer0_output.detach()
        self.feature[1] = layer1_output.detach()
        self.feature[2] = layer2_output.detach()
        self.feature[3] = output.detach()

        return output



class MLP_feature(nn.Module):
    def __init__(self, input_zise, H_1, H_2, H_3, d):
        super(MLP_feature, self).__init__()

        self.feature = [[], [], [], [], []]
        self.fc1 = nn.Linear(input_zise, H_1, bias = False)
        self.fc2 = nn.Linear(H_1, H_2, bias = False)
        self.fc3 = nn.Linear(H_2, H_3, bias = False)
        self.fc4 = nn.Linear(H_3, 10, bias = False)
        self.dropout = nn.Dropout(d)

    def forward(self, x):
        layer0_output = x.view(x.shape[0], -1)
        layer1_output = F.relu(self.fc1(layer0_output))
        dropout_output = self.dropout(layer1_output)
        layer2_output = F.relu(self.fc2(dropout_output))
        layer3_output = F.relu(self.fc3(layer2_output))
        output = self.fc4(layer3_output)

        self.feature[0] = layer0_output.detach()
        self.feature[1] = layer1_output.detach()
        self.feature[2] = layer2_output.detach()
        self.feature[3] = layer3_output.detach()
        self.feature[4] = output.detach()

        return output

class FC_feature_fdata(nn.Module):
    def __init__(self, H, d):
        super(FC_feature_fdata, self).__init__()

        self.feature = [[], [], [], []]
        self.fc1 = nn.Linear(500, H, bias = False)
        self.fc2 = nn.Linear(H, H, bias = False)
        self.fc3 = nn.Linear(H, 20, bias = False)
        self.dropout = nn.Dropout(d)

    def forward(self, x):
        layer0_output = x.view(x.shape[0], -1)
        layer1_output = F.relu(self.fc1(layer0_output))

        dropout_output = self.dropout(layer1_output)

        layer2_output = F.relu(self.fc2(dropout_output))

        output = self.fc3(layer2_output)

        self.feature[0] = layer0_output.detach()
        self.feature[1] = layer1_output.detach()
        self.feature[2] = layer2_output.detach()

        self.feature[3] = output.detach()

        return output
    



cfg = [32, 'M', 64, 'M', 128, 128, 'M']

class CNN(nn.Module):
    def __init__(self, num_classes=10, c=3):
        super(CNN, self).__init__()
        self.c = c
        self.features = self._make_layers(cfg)
        self.fc = nn.Linear(128, 20, bias=False)
        self.classifier = nn.Linear(20, num_classes)
        self.feature = [[],[],[],[],[],[],[],[],[]]
        

    def forward(self, x):
        x = self.features(x)
        x = nn.AdaptiveAvgPool2d((1, 1))(x)
        fc_feature = torch.flatten(x, 1)
        x = self.fc(fc_feature)
        x = F.relu(x)
        x = self.classifier(x)

        self.feature[8] = fc_feature.detach()
        return x

    def _make_layers(self, cfg):
        layers = []
        in_channels = self.c  
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [
                    nn.Conv2d(in_channels, v, kernel_size=3, padding=1),
                    # nn.BatchNorm2d(v),
                    nn.ReLU(inplace=True)
                ]
                in_channels = v
        return nn.Sequential(*layers)    



