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


class FC_feature_multilayer(nn.Module):
    def __init__(self, input_zise, hidden_sizes, d, num_classes=10):
        super(FC_feature_multilayer, self).__init__()

        if isinstance(hidden_sizes, int):
            hidden_sizes = [hidden_sizes]
        if len(hidden_sizes) == 0:
            raise ValueError("hidden_sizes should contain at least one layer size.")

        self.hidden_layers = nn.ModuleList()
        in_dim = input_zise
        for h in hidden_sizes:
            self.hidden_layers.append(nn.Linear(in_dim, h, bias=False))
            in_dim = h

        self.classifier = nn.Linear(in_dim, num_classes, bias=False)
        self.dropout = nn.Dropout(d)

        # feature[0] is flattened input, feature[1..n] are hidden activations,
        # feature[-1] is logits.
        self.feature = [[] for _ in range(len(hidden_sizes) + 2)]

    def forward(self, x):
        layer_output = x.view(x.shape[0], -1)
        self.feature[0] = layer_output.detach()

        for i, layer in enumerate(self.hidden_layers):
            layer_output = F.relu(layer(layer_output))
            self.feature[i + 1] = layer_output.detach()

            # Keep behavior consistent with FC_feature: dropout between hidden layers.
            if i < len(self.hidden_layers) - 1:
                layer_output = self.dropout(layer_output)

        output = self.classifier(layer_output)
        self.feature[-1] = output.detach()

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
    

cfg =  [32, 'M', 32, 32, 'M'] 

class CNN_ln(nn.Module):
    def __init__(self, num_classes=10):
        super(CNN_ln, self).__init__()
        self.features = self._make_layers(cfg)
        self.fc = nn.Linear(32, 32, bias=False)
        self.classifier = nn.Linear(32, num_classes)
        self.feature = [[],[],[],[],[],[],[],[],[]]
        self.ln = nn.LayerNorm(32)

    def forward(self, x):
        x = self.features(x)
        x = nn.AdaptiveAvgPool2d((1, 1))(x)
        x = torch.flatten(x, 1)
        x = self.ln(x)
        fc_feature = F.relu(x)
        x = self.fc(fc_feature)
        x = F.relu(x)
        x = self.classifier(x)

        self.feature[8] = fc_feature.detach()
        return x

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
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






# BasicBlock for CIFAR-ResNet
class BasicBlock(nn.Module):
    expansion = 1
    
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        
        self.shortcut = nn.Sequential()
        # When dimensions change, use 1x1 conv
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


# ResNet-CIFAR
class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 16
        
        # First layer: 3x3 conv (not 7x7 like ImageNet)
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        
        # 3 stages: each stage has n blocks (n=3 for ResNet-20)
        self.layer1 = self._make_layer(block, 16,  num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32,  num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64,  num_blocks[2], stride=2)
        
        self.linear = nn.Linear(64, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet20():
    return ResNet(BasicBlock, [3, 3, 3])
