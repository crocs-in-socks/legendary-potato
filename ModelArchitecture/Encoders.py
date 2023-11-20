import torch
import torch.nn as nn
import torch.nn.functional as F

class ResNet3D_Decoder(nn.Module):
    def __init__(self, feature_channels, order=5):
        super().__init__()

        self.order = order
        # upsample = nn.Sequential(
        #     nn.ConvTranspose3d(feature_channels, feature_channels, kernel_size=2, stride=2),
        #     nn.BatchNorm3d(feature_channels)
        # )
        # upsample = nn.Sequential(
        #     nn.Upsample(scale_factor=2, mode='trilinear'),
        #     nn.BatchNorm3d(feature_channels)
        # )
        self.layers = [
            nn.ConvTranspose3d(feature_channels, feature_channels, kernel_size=2, stride=2),
            # nn.BatchNorm3d(feature_channels)
        ] * self.order
        # self.layers = [
        #     nn.Upsample(scale_factor=4, mode='trilinear'),
        #     nn.BatchNorm3d(feature_channels)
        # ] * self.order
        # self.layers = [
        #     nn.Upsample(scale_factor=4, mode='trilinear'),
        #     nn.BatchNorm3d(feature_channels),
        #     nn.Upsample(scale_factor=4, mode='trilinear'),
        #     nn.BatchNorm3d(feature_channels),
        #     nn.Upsample(scale_factor=2, mode='trilinear'),
        #     nn.BatchNorm3d(feature_channels)
        # ]

        # self.upsampler = nn.Upsample(size=(128, 128, 128), mode='trilinear')

        self.upsampler = nn.Sequential(*self.layers)
    
    def forward(self, x):
        x = self.upsampler(x)
        return x

class ResNet3D_EncoderBasicBlock(nn.Module):
    def __init__(self, input_channels, output_channels, stride=1, expansion=1, downsample=None):
        super().__init__()
        self.expansion = expansion
        self.downsample = downsample

        self.conv1 = nn.Conv3d(input_channels, output_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm3d(output_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv3d(output_channels, output_channels*self.expansion, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(output_channels*self.expansion)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample:
            identity = self.downsample(x)
            
        out += identity
        out = self.relu(out)
        return out
    
class ResNet3D_Encoder(nn.Module):
    def __init__(self, image_channels):
        super().__init__()
        self.expansion = 1
        self.in_channels = 64

        self.conv1 = nn.Conv3d(image_channels, self.in_channels, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm3d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(ResNet3D_EncoderBasicBlock, 64, 2)
        self.layer2 = self._make_layer(ResNet3D_EncoderBasicBlock, 128, 2, stride=2)
        self.layer3 = self._make_layer(ResNet3D_EncoderBasicBlock, 256, 2, stride=2)
        self.layer4 = self._make_layer(ResNet3D_EncoderBasicBlock, 512, 2, stride=2)

    def _make_layer(self, block, output_channels, blocks, stride=1):
        downsample = None
        if stride != 1:
            downsample = nn.Sequential(
                nn.Conv3d(self.in_channels, output_channels*self.expansion, kernel_size=1, stride=stride),
                nn.BatchNorm3d(output_channels*self.expansion),
            )
        
        layers = []
        layers.append(
            ResNet3D_EncoderBasicBlock(
                self.in_channels,
                output_channels, 
                stride,
                self.expansion,
                downsample
            )
        )
        self.in_channels = output_channels * self.expansion

        for _ in range(1, blocks):
            layers.append(
                ResNet3D_EncoderBasicBlock(
                    self.in_channels,
                    output_channels,
                    expansion=self.expansion
                )
            )
        
        return nn.Sequential(*layers)
    
    def forward(self, x):  
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)

        out1 = self.layer1(out)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)

        # out_dict = {
        #         'x': x,
        #         'out': out,
        #         'out1': out1,
        #         'out2': out2,
        #         'out3': out3,
        #         'out4': out4
        # }

        # return out_dict
        # return out3
        return out4

class VGG3D(nn.Module):

    def __init__(self, input_channels, output_classes):
        super().__init__()

        self.enc_layer1 = nn.Sequential(
            nn.Conv3d(input_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(num_features=64),
            nn.ReLU(inplace=True)
        )

        self.enc_layer2 = nn.Sequential(
            nn.Conv3d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(num_features=64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2)
        )

        self.enc_layer3 = nn.Sequential(
            nn.Conv3d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(num_features=128),
            nn.ReLU(inplace=True)
        )

        self.enc_layer4 = nn.Sequential(
            nn.Conv3d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(num_features=128),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size = 2, stride = 2)
        )

        self.enc_layer5 = nn.Sequential(
            nn.Conv3d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(num_features=256),
            nn.ReLU(inplace=True)
        )

        self.enc_layer6 = nn.Sequential(
            nn.Conv3d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(num_features=256),
            nn.ReLU(inplace=True)
        )

        self.enc_layer7 = nn.Sequential(
            nn.Conv3d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(num_features=256),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size = 2, stride = 2)
        )

        self.enc_layer8 = nn.Sequential(
            nn.Conv3d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(num_features=512),
            nn.ReLU(inplace=True)
        )

        self.enc_layer9 = nn.Sequential(
            nn.Conv3d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(num_features=512),
            nn.ReLU(inplace=True)
        )

        self.enc_layer10 = nn.Sequential(
            nn.Conv3d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(num_features=512),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size = 2, stride = 2)
        )

        self.enc_layer11 = nn.Sequential(
            nn.Conv3d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(num_features=512),
            nn.ReLU(inplace=True)
        )

        self.enc_layer12 = nn.Sequential(
            nn.Conv3d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(512),
            nn.ReLU(inplace=True)
        )

        self.proj_layer1 = nn.Sequential(
            nn.Linear(4*4*4*512, 4096),
            nn.ReLU(inplace=True)
        )

        self.proj_layer2 = nn.Sequential(
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True)
        )

        self.encoder = nn.Sequential()
        self.encoder.add_module('enc-layer1', self.enc_layer1) # input -> 64
        self.encoder.add_module('enc-layer2', self.enc_layer2) # 64 -> 64
        self.encoder.add_module('enc-layer3', self.enc_layer3) # 64 -> 128
        self.encoder.add_module('enc-layer4', self.enc_layer4) # 128 -> 128
        self.encoder.add_module('enc-layer5', self.enc_layer5) # 128 -> 256
        self.encoder.add_module('enc-layer6', self.enc_layer6) # 256 -> 256
        self.encoder.add_module('enc-layer7', self.enc_layer7) # 256 -> 256

        # self.encoder.add_module('enc-layer8', self.enc_layer8) # 256 -> 512
        # self.encoder.add_module('enc-layer9', self.enc_layer9) # 512 -> 512
        # self.encoder.add_module('enc-layer10', self.enc_layer10) # 512 -> 512
        # self.encoder.add_module('enc-layer11', self.enc_layer11) # 512 -> 512
        # self.encoder.add_module('enc-layer12', self.enc_layer12) # 512 -> 512

        # self.projection_head = nn.Sequential()
        # self.projection_head.add_module('proj-layer1', self.proj_layer1)
        # self.projection_head.add_module('proj-layer2', self.proj_layer2)

    def forward(self, x):
        out = self.encoder(x)
        # print(f'Encoder output: {out.shape}')
        # out = torch.reshape(out, shape=(out.shape[0], -1))
        # print(f'Projection head input: {out.shape}')

        # out = self.projection_head(out)
        return out

class Classifier(nn.Module):

    def __init__(self, input_channels, output_channels):
        super().__init__()

        self.layer1 = nn.Sequential(
            nn.Linear(input_channels, input_channels // 4),
            nn.ReLU(inplace=True)
        )

        self.layer2 = nn.Sequential(
            nn.Linear(input_channels // 4, input_channels // (4*4)),
            nn.ReLU(inplace=True)
        )

        self.layer3 = nn.Sequential(
            nn.Linear(input_channels // (4*4), input_channels // (4*4*4)),
            nn.ReLU(inplace=True)
        )

        self.layer4 = nn.Linear(input_channels // (4*4*4), output_channels)

        self.fc = nn.Sequential()
        self.fc.add_module('layer1', self.layer1)
        self.fc.add_module('layer2', self.layer2)
        self.fc.add_module('layer3', self.layer3)
        self.fc.add_module('layer4', self.layer4)

        self.activation = nn.Sigmoid()
    
    def forward(self, x):
        out = self.fc(x)
        out = self.activation(out)
        return out

class WideResNet3D_BasicBlock(nn.Module):
    def __init__(self, input_channels, channels, dropout_rate, stride=1):
        super().__init__()
        self.bn1 = nn.BatchNorm3d(input_channels)
        self.conv1 = nn.Conv3d(input_channels, channels, kernel_size=3, padding=1)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.bn2 = nn.BatchNorm3d(channels)
        self.conv2 = nn.Conv3d(channels, channels, kernel_size=3, stride=stride, padding=1)

        self.shortcut = nn.Sequential()
        if stride != 1 or input_channels != channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(input_channels, channels, kernel_size=1, stride=stride)
            )
    
    def forward(self, x):
        out = self.dropout(self.conv1(F.relu(self.bn1(x))))
        out = self.conv2(F.relu(self.bn2(out)))
        out += self.shortcut(x)

        return out
    
class WideResNet3d(nn.Module):
    def __init__(self, depth, widen_factor, dropout_rate):
        super().__init__()

        self.input_channels = 64

        # assert ((depth-4)%6 == 0), 'WideResNet depth should be 6n+4'
        # n = (depth-4)/6
        n = depth
        k = widen_factor

        nStages = [16, 16*k, 32*k, 64*k]

        self.conv1 = nn.Conv3d(1, nStages[0], kernel_size=3, stride=1, padding=1)
        self.layer1 = self._wide_layer(WideResNet3D_BasicBlock, nStages[1], n, dropout_rate, stride=1)
        self.layer2 = self._wide_layer(WideResNet3D_BasicBlock, nStages[2], n, dropout_rate, stride=2)
        self.layer3 = self._wide_layer(WideResNet3D_BasicBlock, nStages[3], n, dropout_rate, stride=2)
        self.bn1 = nn.BatchNorm3d(nStages[3], momentum=0.9)
        
    def _wide_layer(self, block, channels, num_blocks, dropout_rate, stride):
        strides = [stride] + [1]*(int(num_blocks)-1)
        layers = []

        for stride in strides:
            layers.append(block(self.input_channels, channels, dropout_rate, stride))
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        # out = F.relu(self.bn1(out))
        # out = F.avg_pool3d(out, 8)

        return out