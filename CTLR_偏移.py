import torch
import torch.nn as nn
import torch.nn.functional as F

class SmallPatchAttention(nn.Module):
    """专门针对小裁块设计的2D注意力机制"""
    def __init__(self, channels, patch_size=16):
        super(SmallPatchAttention, self).__init__()
        self.patch_size = patch_size
        self.local_attention = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels),
            nn.BatchNorm2d(channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=1)
        )
        
        # 轻量级通道注意力
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels//4, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(channels//4, channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        local_attn = self.local_attention(x)
        channel_attn = self.channel_attention(x)
        return local_attn * channel_attn

class CompactConvBlock2D(nn.Module):
    """针对小裁块优化的紧凑型2D卷积块"""
    def __init__(self, in_channels, out_channels):
        super(CompactConvBlock2D, self).__init__()
        
        # 使用小卷积核和分组卷积减少参数量
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, groups=1),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )
        
        # 深度可分离卷积
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, groups=out_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )
        
        self.small_patch_attention = SmallPatchAttention(out_channels)
        
        # 轻量级残差连接
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels)
        ) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.small_patch_attention(x)
        return x + identity

class SmallPatchUnet2D(nn.Module):
    """专门针对小裁块优化的2D U-Net"""
    def __init__(self, in_channels=1, out_channels=1, channels=[32, 64, 128]):
        super(SmallPatchUnet2D, self).__init__()
        
        self.channels = channels
        self.num_layers = len(self.channels) - 1

        # 轻量级输入编码器
        self.incoder = nn.Sequential(
            nn.Conv2d(in_channels, channels[0]//2, kernel_size=3, padding=1),
            nn.GELU()
        )

        # 编码器
        self.encoders = nn.ModuleList([
            nn.Sequential(
                CompactConvBlock2D(
                    channels[i-1] if i > 0 else channels[0]//2,
                    channels[i]
                ),
                nn.MaxPool2d(kernel_size=2, stride=2)
            ) for i in range(len(channels))
        ])

        # 解码器
        self.decoders = nn.ModuleList([
            nn.Sequential(
                nn.ConvTranspose2d(
                    channels[i], channels[i-1],
                    kernel_size=2, stride=2
                ),
                CompactConvBlock2D(
                    channels[i], channels[i-1]
                )
            ) for i in range(len(channels)-1, 0, -1)
        ])

        # 轻量级输出层
        self.outcoder = nn.Sequential(
            nn.Conv2d(channels[0], out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels)
        )
        
        self.fc = None

    def _create_fc(self, input_features, device):
        """创建或更新fc层，输出2个值表示(x,y)偏移"""
        return nn.Sequential(
            nn.Linear(input_features, input_features//4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(input_features//4, 2)  # 输出x,y偏移
        ).to(device)

    def load_state_dict(self, state_dict, strict=True):
        fc_keys = [k for k in state_dict.keys() if k.startswith('fc.')]
        if fc_keys:
            in_features = state_dict['fc.0.weight'].size(1)
            self.fc = self._create_fc(in_features, 'cuda' if torch.cuda.is_available() else 'cpu')
        return super().load_state_dict(state_dict, strict)

    def forward(self, x):
        x = self.incoder(x)
        skips = [x]
        
        # 编码过程
        for encoder in self.encoders:
            x = encoder(x)
            skips.append(x)
        
        # 解码过程
        for i, decoder in enumerate(self.decoders):
            skip = skips[-(i+2)]
            x = decoder[0](x)  # 上采样
            
            if x.size()[2:] != skip.size()[2:]:
                x = F.interpolate(x, size=skip.size()[2:], mode='bilinear', align_corners=True)
            
            x = torch.cat([x, skip], dim=1)
            x = decoder[1](x)
        
        x = self.outcoder(x)
        
        # 获取特征维度并创建fc层
        input_features = x.size(1) * x.size(2) * x.size(3)
        if self.fc is None:
            self.fc = self._create_fc(input_features, x.device)
        
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        return x

def create_model(in_channels=1, out_channels=1, channels=[16, 32, 64], *args):
    return SmallPatchUnet2D(in_channels=in_channels, out_channels=out_channels, 
                           channels=channels)
