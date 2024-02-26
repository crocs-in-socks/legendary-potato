import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

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
        self.in_channels = 16

        self.conv1 = nn.Conv3d(image_channels, self.in_channels, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm3d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(ResNet3D_EncoderBasicBlock, 16, 2)
        self.layer2 = self._make_layer(ResNet3D_EncoderBasicBlock, 32, 2, stride=2)
        self.layer3 = self._make_layer(ResNet3D_EncoderBasicBlock, 64, 2, stride=2)
        self.layer4 = self._make_layer(ResNet3D_EncoderBasicBlock, 128, 2, stride=2)

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

        out_dict = {
                'x': x,
                'out': out,
                'out1': out1,
                'out2': out2,
                'out3': out3,
                'out4': out4
        }

        layer_list = [
            out1,
            out2,
            out3,
            out4
        ]

        final_out = out4

        return layer_list, final_out
        # return out3
        # return out4

class VGG3D_Encoder(nn.Module):

    def __init__(self, input_channels):
        super().__init__()

        self.enc_layer1 = nn.Sequential(
            nn.Conv3d(input_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(num_features=64),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(num_features=64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2)
        )

        self.enc_layer2 = nn.Sequential(
            nn.Conv3d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(num_features=128),
            nn.ReLU(inplace=True),
            nn.Conv3d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(num_features=128),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size = 2, stride = 2)
        )

        self.enc_layer3 = nn.Sequential(
            nn.Conv3d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(num_features=256),
            nn.ReLU(inplace=True),
            nn.Conv3d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(num_features=256),
            nn.ReLU(inplace=True)
        )

        self.enc_layer4 = nn.Sequential(
            nn.Conv3d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(num_features=512),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size = 2, stride = 2),
            nn.Conv3d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(num_features=512),
            nn.ReLU(inplace=True)
        )

        # self.enc_layer5 = nn.Sequential(
            # nn.Conv3d(256, 512, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm3d(num_features=512),
            # nn.ReLU(inplace=True),
            # nn.Conv3d(512, 512, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm3d(num_features=512),
            # nn.ReLU(inplace=True),
            # nn.MaxPool3d(kernel_size = 2, stride = 2)
        # )

    def forward(self, x):
        out1 = self.enc_layer1(x)
        out2 = self.enc_layer2(out1)
        out3 = self.enc_layer3(out2)
        out4 = self.enc_layer4(out3)
        # out5 = self.enc_layer5(out4)

        layer_list = [
            out1,
            out2,
            out3,
            out4,
            # out5
        ]

        final_out = out4

        return layer_list, final_out
    
class SA_UNet_Encoder(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, init_features=64):
        super().__init__()

        features = init_features
        self.encoder1 = SA_UNet_Encoder._block(in_channels, features, name="enc1")
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.encoder2 = SA_UNet_Encoder._block(features, features * 2, name="enc2")
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.encoder3 = SA_UNet_Encoder._block(features * 2, features * 4, name="enc3")
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.encoder4 = SA_UNet_Encoder._block(features * 4, features * 8, name="enc4")
        self.pool4 = nn.MaxPool3d(kernel_size=2, stride=2)

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        layer_list = [
            enc1,
            enc2,
            enc3,
            enc4
        ]

        final_out = enc4

        return layer_list, final_out

    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv3d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=True,
                        ),
                    ),
                    (name + "drop1", nn.Dropout3d(p=0.15,inplace=True)),
                    (name + "relu1", nn.ReLU(inplace=True)),
                    (name + "norm1", nn.BatchNorm3d(num_features=features)),

                    (
                        name + "conv2",
                        nn.Conv3d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=True,
                        ),
                    ),
                    (name + "drop2", nn.Dropout3d(p=0.15,inplace=True)),
                    (name + "relu2", nn.ReLU(inplace=True)),
                    (name + "norm2", nn.BatchNorm3d(num_features=features)),
                ]
            )
        )

class Classifier(nn.Module):

    def __init__(self, input_channels, output_channels, pooling_size=2):
        super().__init__()

        self.pooler = nn.AdaptiveAvgPool3d((pooling_size, pooling_size, pooling_size))

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

        out = self.pooler(x)
        out = torch.reshape(out, shape=(out.shape[0], -1))
        out = self.fc(out)
        out = self.activation(out)
        return out
    
class Projector(nn.Module):
    def __init__(self, num_layers, layer_sizes, test=False):
        super().__init__()
        self.num_layers = num_layers
        self.projection_heads = nn.ModuleList([nn.Conv3d(layer_sizes[idx], 1, kernel_size=1) for idx in range(num_layers)])
        self.super_projection_head = nn.Conv3d(num_layers, 1, kernel_size=1)
        self.test = test
    
    def forward(self, x):
        individual_out = [self.projection_heads[idx](x[idx]) for idx in range(self.num_layers)]
        largest_layer_shape = individual_out[0].shape[-1]
        upsampled_out = [F.interpolate(individual_out[idx], size=(largest_layer_shape, largest_layer_shape, largest_layer_shape), mode='trilinear') for idx in range(self.num_layers)]
        stacked_out = torch.cat(upsampled_out, dim=1)
        out = self.super_projection_head(stacked_out)

        if self.test:
            return out, upsampled_out
        return out
    
class IntegratedChannelProjector(nn.Module):
    def __init__(self, num_layers, layer_sizes, layer_dimensions, reduction_ratio=16):
        super().__init__()
        self.num_layers = num_layers

        self.average_pool_heads = nn.ModuleList([
            nn.Sequential(
                nn.AvgPool3d(kernel_size=(layer_dimensions[idx], layer_dimensions[idx], layer_dimensions[idx]), stride=(layer_dimensions[idx], layer_dimensions[idx], layer_dimensions[idx])),
                nn.Flatten(),
                nn.Linear(layer_sizes[idx], layer_sizes[idx] // reduction_ratio),
                nn.ReLU(inplace=True),
                nn.Linear(layer_sizes[idx] // reduction_ratio, layer_sizes[idx])
            )
            for idx in range(num_layers)
        ])

        self.max_pool_heads = nn.ModuleList([
            nn.Sequential(
                nn.MaxPool3d(kernel_size=(layer_dimensions[idx], layer_dimensions[idx], layer_dimensions[idx]), stride=(layer_dimensions[idx], layer_dimensions[idx], layer_dimensions[idx])),
                nn.Flatten(),
                nn.Linear(layer_sizes[idx], layer_sizes[idx] // reduction_ratio),
                nn.ReLU(inplace=True),
                nn.Linear(layer_sizes[idx] // reduction_ratio, layer_sizes[idx])
            )
            for idx in range(num_layers)
        ])
    
    def forward(self, x):
        return [F.sigmoid(self.average_pool_heads[idx](x[idx]) + self.max_pool_heads[idx](x[idx])) for idx in range(self.num_layers)]

class IntegratedSpatialProjector(nn.Module):
    def __init__(self, num_layers, layer_sizes, projected_channels=None):
        super().__init__()
        self.num_layers = num_layers

        if projected_channels is None:
            # projected_channels = [1]*self.num_layers
            projected_channels = layer_sizes

        # self.projection_heads = nn.ModuleList([
        #         nn.Sequential(
        #             nn.Conv3d(layer_sizes[idx], 1, kernel_size=1),
        #             nn.ReLU(inplace=True),
        #             nn.Conv3d(1, projected_channels[idx], kernel_size=1)
        #         ) for idx in range(num_layers)
        #     ])

        self.projection_heads = nn.ModuleList([
                nn.Sequential(
                    nn.Conv3d(layer_sizes[idx], layer_sizes[idx] // 4, kernel_size=1),
                    nn.ReLU(inplace=True),
                    nn.Conv3d(layer_sizes[idx] // 4, projected_channels[idx], kernel_size=1)
                ) for idx in range(num_layers)
            ])
    
    def forward(self, x):
        return [self.projection_heads[idx](x[idx]) for idx in range(self.num_layers)]
    
class DUCKproxy_Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.up1 = nn.ConvTranspose3d(272, 136, kernel_size=2, stride=2)
        self.up2 = nn.ConvTranspose3d(136, 68, kernel_size=2, stride=2)
        self.up3 = nn.ConvTranspose3d(68, 34, kernel_size=2, stride=2)
        self.up4 = nn.ConvTranspose3d(34, 17, kernel_size=2, stride=2)
        self.up5 = nn.ConvTranspose3d(17, 1, kernel_size=2, stride=2)
    
    def forward(self, x):
        out = self.up1(x)
        out = self.up2(out)
        out = self.up3(out)
        out = self.up4(out)
        out = self.up5(out)
        
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