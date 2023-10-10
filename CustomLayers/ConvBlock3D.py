import torch
import torch.nn as nn


kernel_initializer = 'he_uniform'

class Gen_Conv3d_Block(nn.Module):
    def __init__(self,filters, block_type, dilation=1, size=3, padding='same', repeat=1):
        self.block_type = block_type
        self.filters =  filters
        self.block_dict = {'seperated':Seperated_Conv3d_Block, 'duck':Duck_Conv3d_Block, 'midscope': Midscope_Conv3d_Block, 
                           'widescope':Widescope_Conv3d_Block, 'resnet':Resnet_Conv3d_Block,'double_convolution':DoubleConv_with_BatchNorm, 'conv':Conv3d_Block}
        self.conv_block = nn.Sequential([self.block_dict[self.block_type(filters,dilation,size,padding)]*repeat])

    def forward(self,x):
        out = self.conv_block(x)
        return out

class Conv3d_Block(nn.Module):
    def __init__(self,filters,size=3,padding='same',dilation_rate=1):
        self.conv = nn.Conv3d(filters, filters, (size, size, size), padding='same')
    def forward(self,x):
        out = self.conv(x)

class Duck_Conv3d_Block(nn.Mconv_block_2Dodule):
    def __init__(self,filters,size=3,padding='same',dilation_rate=1):
        self.bn1 = nn.BatchNorm3d(filters)
        self.wide = Widescope_Conv3d_Block(filters)
        self.mid = Midscope_Conv3d_Block(filters)
        
        self.res_1 = Resnet_Conv3d_Block(filters)
        
        self.res_2_1 = Resnet_Conv3d_Block(filters)
        self.res_2_2 = Resnet_Conv3d_Block(filters)

        self.res_3_1 = Resnet_Conv3d_Block(filters)
        self.res_3_2 = Resnet_Conv3d_Block(filters)
        self.res_3_3 = Resnet_Conv3d_Block(filters)

        self.sep = Seperated_Conv3d_Block(filters,size=6)
        
        self.bn2 = nn.BatchNorm3d(filters)

    def forward(self,x):
        out = self.bn1(x)

        out_wide = self.wide(out)
        out_mid = self.wide(out)
        
        out_res_1 = self.res_1(out)
        
        out_res_2 = self.res_2_1(out)
        out_res_2 = self.res_2_2(out_res_2)

        out_res_3 = self.res_3_1(out)
        out_res_3 = self.res_3_2(out_res_3)
        out_res_3 = self.res_3_3(out_res_3)

        out_sep = self.sep(out)

        out  = out_wide + out_mid + out_res_1 + out_res_2 + out_res_3 + out_sep

        out = self.wide(out)
        
        return out

class Seperated_Conv3d_Block(nn.Module):
    def __init__(self,filters,size=3,padding='same',dilation_rate=1):
        self.conv1 = nn.Conv3d(filters, filters, (1, 1, size), padding='same')
        self.relu1 = nn.ReLU()
        self.bn1 = nn.BatchNorm3d(filters)

        self.conv2 = nn.Conv3d(filters, filters, (1, size, 1), padding='same')
        self.relu2 = nn.ReLU()
        self.bn2 = nn.BatchNorm3d(filters)

        self.conv2 = nn.Conv3d(filters, filters, (size, 1, 1), padding='same')
        self.relu2 = nn.ReLU()
        self.bn2 = nn.BatchNorm3d(filters)

    def forward(self,x):
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.bn1(out)
                         
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.bn2(out)

        return out        

class Midscope_Conv3d_Block(nn.Module):
    def __init__(self,filters,size=3,padding='same',dilation_rate=1):
        self.conv1 = nn.Conv3d(filters, filters, (3, 3, 3), padding='same',dilation_rate=1)
        self.relu1 = nn.ReLU()
        self.bn1 = nn.BatchNorm3d(filters)

        self.conv2 = nn.Conv3d(filters, filters, (3, 3, 3), padding='same', dilation_rate=2)
        self.relu2 = nn.ReLU()
        self.bn2 = nn.BatchNorm3d(filters)

    def forward(self,x):
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.bn1(out)
                         
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.bn2(out)

        return out        

class Widescope_Conv3d_Block(nn.Module):
    def __init__(self,filters,size=3,padding='same',dilation_rate=1):
        self.conv1 = nn.Conv3d(filters, filters, (3, 3, 3), padding='same',dilation_rate=1)
        self.relu1 = nn.ReLU()
        self.bn1 = nn.BatchNorm3d(filters)

        self.conv2 = nn.Conv3d(filters, filters, (3, 3, 3), padding='same', dilation_rate=2)
        self.relu2 = nn.ReLU()
        self.bn2 = nn.BatchNorm3d(filters)

        self.conv3 = nn.Conv3d(filters, filters, (3, 3, 3), padding='same', dilation_rate=3)
        self.relu3 = nn.ReLU()
        self.bn3 = nn.BatchNorm3d(filters)

    def forward(self,x):
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.bn1(out)
                         
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.bn2(out)

        out = self.conv3(out)
        out = self.relu3(out)
        out = self.bn3(out)
        return out        

class Resnet_Conv3d_Block(nn.Module):
    def __init__(self,filters,size=3,padding='same',dilation_rate=1):
        self.conv_res = nn.Conv3d(filters, filters, (1, 1, 1), kernel_initializer=kernel_initializer, padding='same',
               dilation_rate=dilation_rate)
        
        self.conv1 = nn.Conv3d(filters, filters, (3, 3, 3), kernel_initializer=kernel_initializer, padding='same',
               dilation_rate=dilation_rate)
        self.relu1 = nn.ReLU()
        self.bn1 = nn.BatchNorm3d(filters)  
        self.conv2 = nn.Conv3d(filters, filters, (3, 3, 3), kernel_initializer=kernel_initializer, padding='same',
               dilation_rate=dilation_rate)
        self.relu2 = nn.ReLU()
        self.bn2 = nn.BatchNorm3d(filters)  

        self.bn3 = nn.BatchNorm3d(filters)

    def forward(self,x):
        out_res = self.conv_res(x)

        out = self.conv1(x)
        out = self.relu1(out)
        out = self.bn1(out)
                         
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.bn2(out)
        
        out = out + out_res
        out = self.bn3(out)

        return out        

class DoubleConv_with_BatchNorm(nn.Module):
    def __init__(self,filters,size=3,padding='same',dilation_rate=1):
        self.conv1 = nn.Conv3d(filters, filters, (3, 3, 3), padding='same',
               dilation_rate=dilation_rate)
        self.relu1 = nn.ReLU()
        self.bn1 = nn.BatchNorm3d(filters)  
        self.conv2 = nn.Conv3d(filters, filters, (3, 3, 3), padding='same',
               dilation_rate=dilation_rate)
        self.relu2 = nn.ReLU()
        self.bn2 = nn.BatchNorm3d(filters)  

    def forward(self,x):
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.bn1(out)
                         
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.bn2(out)
        return out        


# def conv_block_2D(x, filters, block_type, repeat=1, dilation_rate=1, size=3, padding='same'):
#     result = x

#     for i in range(0, repeat):

#         if block_type == 'separated':
#             result = separated_conv2D_block(result, filters, size=size, padding=padding)
#         elif block_type == 'duckv2':
#             result = duckv2_conv2D_block(result, filters, size=size)
#         elif block_type == 'midscope':
#             result = midscope_conv2D_block(result, filters)
#         elif block_type == 'widescope':
#             result = widescope_conv2D_block(result, filters)
#         elif block_type == 'resnet':
#             result = resnet_conv2D_block(result, filters, dilation_rate)
#         elif block_type == 'conv':
#             result = Conv2D(filters, (size, size),
#                             activation='relu', kernel_initializer=kernel_initializer, padding=padding)(result)
#         elif block_type == 'double_convolution':
#             result = double_convolution_with_batch_normalization(result, filters, dilation_rate)

#         else:
#             return None
#     return result


# def duckv2_conv2D_block(x, filters, size):
#     x = BatchNormalizationV2(axis=-1)(x)
#     x1 = widescope_conv2D_block(x, filters)

#     x2 = midscope_conv2D_block(x, filters)

#     x3 = conv_block_2D(x, filters, 'resnet', repeat=1)

#     x4 = conv_block_2D(x, filters, 'resnet', repeat=2)

#     x5 = conv_block_2D(x, filters, 'resnet', repeat=3)

#     x6 = separated_conv2D_block(x, filters, size=6, padding='same')

#     x = add([x1, x2, x3, x4, x5, x6])

#     x = BatchNormalizationV2(axis=-1)(x)

#     return x

# def separated_conv2D_block(x, filters, size=3, padding='same'):
#     x = Conv2D(filters, (1, size), activation='relu', kernel_initializer=kernel_initializer, padding=padding)(x)

#     x = BatchNormalizationV2(axis=-1)(x)

#     x = Conv2D(filters, (size, 1), activation='relu', kernel_initializer=kernel_initializer, padding=padding)(x)

#     x = BatchNormalizationV2(axis=-1)(x)

#     return x

# def midscope_conv2D_block(x, filters):
#     x = Conv2D(filters, (3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same',
#                dilation_rate=1)(x)

#     x = BatchNormalizationV2(axis=-1)(x)

#     x = Conv2D(filters, (3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same',
#                dilation_rate=2)(x)

#     x = BatchNormalizationV2(axis=-1)(x)

#     return x

# def widescope_conv2D_block(x, filters):
#     x = Conv2D(filters, (3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same',
#                dilation_rate=1)(x)

#     x = BatchNormalizationV2(axis=-1)(x)

#     x = Conv2D(filters, (3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same',
#                dilation_rate=2)(x)

#     x = BatchNormalizationV2(axis=-1)(x)

#     x = Conv2D(filters, (3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same',
#                dilation_rate=3)(x)

#     x = BatchNormalizationV2(axis=-1)(x)

#     return x

# def resnet_conv2D_block(x, filters, dilation_rate=1):
#     x1 = nn.Conv2d(filters, filters, (1, 1), activation='relu', padding='same',dilation_rate=dilation_rate)

#     x = nn.Conv2d(filters, filters, (3, 3), activation='relu', padding='same',dilation_rate=dilation_rate)
#     x = nn.BatchNorm3d(filters) 
#     x = nn.Conv2d(filters, filters, (3, 3), activation='relu', padding='same',dilation_rate=dilation_rate)
#     x = nn.BatchNorm3d(filters) 
#     x_final = add([x, x1])

#     x_final = nn.BatchNorm3d(filters) 

#     return x_final

# def double_convolution_with_batch_normalization(x, filters, dilation_rate=1):
#     x = Conv2D(filters, (3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same',
#                dilation_rate=dilation_rate)(x)
#     x = BatchNormalizationV2(axis=-1)(x)
#     x = Conv2D(filters, (3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same',
#                dilation_rate=dilation_rate)(x)
#     x = BatchNormalizationV2(axis=-1)(x)

#     return x
