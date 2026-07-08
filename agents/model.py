import torch
import torch.nn as nn
import torch.nn.functional as F

class SEBlock(nn.Module):
    """Модуль Squeeze-and-Excitation для оценки глобального контекста доски."""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class ResBlock(nn.Module):
    def __init__(self, num_hidden):
        super().__init__()
        self.conv1 = nn.Conv2d(num_hidden, num_hidden, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_hidden)
        self.conv2 = nn.Conv2d(num_hidden, num_hidden, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_hidden)
        self.se = SEBlock(num_hidden) # Внедряем глобальное внимание

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)            # Применяем SE перед сложением
        out += residual
        return F.relu(out)

class WallzNet(nn.Module):
    # По умолчанию теперь 10 каналов и 10 блоков глубины
    def __init__(self, num_channels=10, num_res_blocks=10, num_hidden=128):
        super().__init__()
        
        self.start_block = nn.Sequential(
            nn.Conv2d(num_channels, num_hidden, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_hidden),
            nn.ReLU()
        )
        
        self.res_blocks = nn.ModuleList(
            [ResBlock(num_hidden) for _ in range(num_res_blocks)]
        )
        
        self.policy_conv = nn.Conv2d(num_hidden, 32, kernel_size=1)
        self.policy_bn = nn.BatchNorm2d(32)
        self.policy_fc = nn.Linear(32 * 9 * 9, 209)
        
        self.value_conv = nn.Conv2d(num_hidden, 3, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(3)
        self.value_fc1 = nn.Linear(3 * 9 * 9, 64)
        self.value_fc2 = nn.Linear(64, 1)

    def forward(self, x, action_mask=None):
        x = self.start_block(x)
        for res_block in self.res_blocks:
            x = res_block(x)
            
        p = F.relu(self.policy_bn(self.policy_conv(x)))
        p = p.view(p.size(0), -1)
        logits = self.policy_fc(p)
        
        if action_mask is not None:
            logits = logits.masked_fill(~action_mask, -1e9)
            
        v = F.relu(self.value_bn(self.value_conv(x)))
        v = v.view(v.size(0), -1)
        v = F.relu(self.value_fc1(v))
        value = torch.tanh(self.value_fc2(v))
        
        return logits, value