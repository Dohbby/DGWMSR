import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class SRGAN(nn.Module):
    def __init__(self, input_height=64, input_width=64, input_channels=1, batch_size=128, kernel_size=(3, 3)):
        super(SRGAN, self).__init__()
        self.input_height = input_height
        self.input_width = input_width
        self.input_channels = input_channels
        self.batch_size = batch_size
        self.kernel_size = kernel_size
        self.num_points = kernel_size[0] * kernel_size[1]
        self.num_channels = input_channels
        self.extend_scope = 2.0

        # Initial convolution
        self.conv1 = nn.Conv2d(input_channels, 64, 3, padding=1)

        # First block
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv4 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv5 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv6 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv6_1 = nn.Conv2d(64, 64, 3, padding=1)
        self.ca_at1 = nn.Sequential(
            nn.Conv2d(256, 256 // 16, 1),
            nn.Conv2d(256 // 16, 256, 1)
        )
        self.conv7 = nn.Conv2d(256, 64, 1)

        # Second block
        self.conv8 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv9 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv10 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv11 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv12 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv12_1 = nn.Conv2d(64, 64, 3, padding=1)
        self.ca_at2 = nn.Sequential(
            nn.Conv2d(256, 256 // 16, 1),
            nn.Conv2d(256 // 16, 256, 1)
        )
        self.conv13 = nn.Conv2d(256, 64, 1)

        # Third block
        self.conv14 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv15 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv16 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv17 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv18 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv18_1 = nn.Conv2d(64, 64, 3, padding=1)
        self.ca_at3 = nn.Sequential(
            nn.Conv2d(256, 256 // 16, 1),
            nn.Conv2d(256 // 16, 256, 1)
        )
        self.conv19 = nn.Conv2d(256, 64, 1)

        # Fourth block
        self.conv20 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv21 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv22 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv23 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv24 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv24_1 = nn.Conv2d(64, 64, 3, padding=1)
        self.ca_at4 = nn.Sequential(
            nn.Conv2d(256, 256 // 16, 1),
            nn.Conv2d(256 // 16, 256, 1)
        )
        self.conv25 = nn.Conv2d(256, 64, 1)

        # Fifth block
        self.conv26 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv27 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv28 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv29 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv30 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv30_1 = nn.Conv2d(64, 64, 3, padding=1)
        self.ca_at5 = nn.Sequential(
            nn.Conv2d(256, 256 // 16, 1),
            nn.Conv2d(256 // 16, 256, 1)
        )
        self.conv31 = nn.Conv2d(256, 64, 1)

        # Final layers
        self.ca_at6 = nn.Sequential(
            nn.Conv2d(384, 384 // 16, 1),
            nn.Conv2d(384 // 16, 384, 1)
        )
        self.conv32 = nn.Conv2d(384, 1, 3, padding=1)
        self.sa_conv32 = nn.Sequential(
            nn.Conv2d(1, 1, 3, stride=2, padding=1),
            nn.ConvTranspose2d(1, 1, 3, stride=2, padding=1, output_padding=1),
            nn.Conv2d(1, 1, 3, padding=1)
        )

        # Deformable convolution layers
        self.deform_offset = nn.Conv2d(64, 64, 3, padding=1)
        self.deform_bn = nn.BatchNorm2d(64)
        self.deform_conv = nn.Conv2d(64, 64, 3, padding=1)
        self.deform_bn_out = nn.BatchNorm2d(64)

    def _coordinate_map(self, offset_field):
        x_offset, y_offset = torch.split(
            offset_field.view(self.batch_size, self.input_height, self.input_width, 2, self.num_points),
            1, dim=3
        )
        x_offset = x_offset.squeeze(3)
        y_offset = y_offset.squeeze(3)

        x_center = torch.arange(self.input_width).repeat(self.input_height).view(self.input_height, self.input_width)
        x_center = x_center.repeat(self.num_points, 1, 1).permute(1, 2, 0)
        x_center = x_center.unsqueeze(0).repeat(self.batch_size, 1, 1, 1)

        y_center = torch.arange(self.input_height).repeat(self.input_width).view(self.input_width, self.input_height).t()
        y_center = y_center.repeat(self.num_points, 1, 1).permute(1, 2, 0)
        y_center = y_center.unsqueeze(0).repeat(self.batch_size, 1, 1, 1)

        x = torch.linspace(-(self.kernel_size[0] - 1) / 2, (self.kernel_size[0] - 1) / 2, self.kernel_size[0])
        y = torch.linspace(-(self.kernel_size[1] - 1) / 2, (self.kernel_size[1] - 1) / 2, self.kernel_size[1])
        x, y = torch.meshgrid(x, y, indexing='ij')
        x_grid = x.flatten().repeat(self.input_height * self.input_width).view(self.input_height, self.input_width, self.num_points)
        y_grid = y.flatten().repeat(self.input_height * self.input_width).view(self.input_height, self.input_width, self.num_points)
        x_grid = x_grid.unsqueeze(0).repeat(self.batch_size, 1, 1, 1)
        y_grid = y_grid.unsqueeze(0).repeat(self.batch_size, 1, 1, 1)

        x = x_center + x_grid + self.extend_scope * x_offset
        y = y_center + y_grid + self.extend_scope * y_offset

        x_new = x.view(self.batch_size, self.input_height, self.input_width, self.kernel_size[0], self.kernel_size[1])
        x_new = x_new.permute(1, 3, 0, 2, 4).reshape(self.kernel_size[0] * self.input_height, self.batch_size, self.kernel_size[1] * self.input_width)
        x_new = x_new.permute(1, 0, 2)

        y_new = y.view(self.batch_size, self.input_height, self.input_width, self.kernel_size[0], self.kernel_size[1])
        y_new = y_new.permute(1, 3, 0, 2, 4).reshape(self.kernel_size[0] * self.input_height, self.batch_size, self.kernel_size[1] * self.input_width)
        y_new = y_new.permute(1, 0, 2)

        return x_new, y_new

    def _bilinear_interpolate(self, input_feature, x, y):
        x = x.view(-1)
        y = y.view(-1)

        x0 = torch.floor(x).long()
        x1 = x0 + 1
        y0 = torch.floor(y).long()
        y1 = y0 + 1

        x0 = torch.clamp(x0, 0, self.input_width - 1)
        x1 = torch.clamp(x1, 0, self.input_width - 1)
        y0 = torch.clamp(y0, 0, self.input_height - 1)
        y1 = torch.clamp(y1, 0, self.input_height - 1)

        input_feature_flat = input_feature.view(-1, self.num_channels)

        dimension_2 = self.input_width
        dimension_1 = self.input_width * self.input_height
        base = torch.arange(self.batch_size) * dimension_1
        repeat = torch.ones(self.num_points * self.input_height * self.input_width).unsqueeze(0).t()
        base = base.view(-1, 1).matmul(repeat.t()).view(-1)
        base_y0 = base + y0 * dimension_2
        base_y1 = base + y1 * dimension_2
        index_a = (base_y0 + x0).long()
        index_b = (base_y1 + x0).long()
        index_c = (base_y0 + x1).long()
        index_d = (base_y1 + x1).long()

        value_a = input_feature_flat[index_a]
        value_b = input_feature_flat[index_b]
        value_c = input_feature_flat[index_c]
        value_d = input_feature_flat[index_d]

        x0_float = x0.float()
        x1_float = x1.float()
        y0_float = y0.float()
        y1_float = y1.float()
        area_a = ((x1_float - x) * (y1_float - y)).unsqueeze(1)
        area_b = ((x1_float - x) * (y - y0_float)).unsqueeze(1)
        area_c = ((x - x0_float) * (y1_float - y)).unsqueeze(1)
        area_d = ((x - x0_float) * (y - y0_float)).unsqueeze(1)

        outputs = value_a * area_a + value_b * area_b + value_c * area_c + value_d * area_d
        outputs = outputs.view(self.batch_size, self.kernel_size[0] * self.input_height, self.kernel_size[1] * self.input_width, self.num_channels)
        return outputs

    def deform_conv(self, inputs, offset):
        x, y = self._coordinate_map(offset)
        deformed_feature = self._bilinear_interpolate(inputs, x, y)
        return deformed_feature

    def deform_conv2d(self, inputs):
        offset = self.deform_offset(inputs)
        offset = torch.tanh(self.deform_bn(offset))
        deformed_feature = self.deform_conv(inputs, offset)
        outputs = self.deform_conv(deformed_feature)
        outputs = F.relu(self.deform_bn_out(outputs))
        return outputs

    def spatial_attention(self, x, sa_module):
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        avg_pool = sa_module(avg_pool)
        avg_pool = torch.sigmoid(avg_pool)
        return x * avg_pool

    def channel_attention(self, input_feature, ca_module):
        avg_pool = torch.mean(input_feature, dim=[2, 3], keepdim=True)
        avg_pool = ca_module(avg_pool)
        avg_pool = torch.sigmoid(avg_pool)
        return input_feature * avg_pool

    def generator(self, input_x):
        # Initial convolution
        conv1 = F.relu(self.conv1(input_x))

        # First block
        conv2 = F.relu(self.conv2(conv1))
        conv3 = self.conv3(conv2)
        conv3 = conv1 + conv3
        conv4 = F.relu(self.conv4(conv3))
        conv5 = self.conv5(conv4)
        conv5 = conv3 + conv5
        conv6 = F.relu(self.conv6(conv5))
        conv6 = self.conv6_1(conv6)
        conv6 = conv6 + conv5
        conv6 = torch.cat([conv6, conv5, conv3, conv1], dim=1)
        at1 = self.channel_attention(conv6, self.ca_at1)
        conv7 = F.relu(self.conv7(at1))
        conv7 = conv7 + conv1

        # Second block
        conv8 = F.relu(self.conv8(conv7))
        conv9 = self.conv9(conv8)
        conv9 = conv9 + conv7
        conv10 = F.relu(self.conv10(conv9))
        conv11 = self.conv11(conv10)
        conv11 = conv11 + conv9
        conv12 = F.relu(self.conv12(conv11))
        conv12 = self.conv12_1(conv12)
        conv12 = conv12 + conv11
        conv12 = torch.cat([conv12, conv11, conv9, conv7], dim=1)
        at2 = self.channel_attention(conv12, self.ca_at2)
        conv13 = F.relu(self.conv13(at2))
        conv13 = conv13 + conv7

        # Third block
        conv14 = F.relu(self.conv14(conv13))
        conv15 = self.conv15(conv14)
        conv15 = conv15 + conv13
        conv16 = F.relu(self.conv16(conv15))
        conv17 = self.conv17(conv16)
        conv17 = conv17 + conv15
        conv18 = F.relu(self.conv18(conv17))
        conv18 = self.conv18_1(conv18)
        conv18 = conv18 + conv17
        conv18 = torch.cat([conv18, conv17, conv15, conv13], dim=1)
        at3 = self.channel_attention(conv18, self.ca_at3)
        conv19 = F.relu(self.conv19(at3))
        conv19 = conv19 + conv13

        # Fourth block
        conv20 = F.relu(self.conv20(conv19))
        conv21 = self.conv21(conv20)
        conv21 = conv21 + conv19
        conv22 = F.relu(self.conv22(conv21))
        conv23 = self.conv23(conv22)
        conv23 = conv23 + conv21
        conv24 = F.relu(self.conv24(conv23))
        conv24 = self.conv24_1(conv24)
        conv24 = conv24 + conv23
        conv24 = torch.cat([conv24, conv23, conv21, conv19], dim=1)
        at4 = self.channel_attention(conv24, self.ca_at4)
        conv25 = F.relu(self.conv25(at4))
        conv25 = conv25 + conv19

        # Fifth block
        conv26 = F.relu(self.conv26(conv25))
        conv27 = self.conv27(conv26)
        conv27 = conv27 + conv25
        conv28 = F.relu(self.conv28(conv27))
        conv29 = self.conv29(conv28)
        conv29 = conv29 + conv27
        conv30 = F.relu(self.conv30(conv29))
        conv30 = self.conv30_1(conv30)
        conv30 = conv30 + conv29
        conv30 = torch.cat([conv27, conv29, conv30, conv25], dim=1)
        at5 = self.channel_attention(conv30, self.ca_at5)
        conv31 = F.relu(self.conv31(at5))
        conv31 = conv31 + conv25

        # Final concatenation and output
        conv32 = torch.cat([conv31, conv25, conv19, conv13, conv7, conv1], dim=1)
        conv32 = self.channel_attention(conv32, self.ca_at6)
        conv32 = self.conv32(conv32)
        conv32 = self.spatial_attention(conv32, self.sa_conv32)
        out = conv32 + input_x
        return out

    def forward(self, input_source):
        fake = self.generator(input_source)
        return fake