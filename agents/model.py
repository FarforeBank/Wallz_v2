import torch
import torch.nn as nn
import torch.nn.functional as F

class WallzNet(nn.Module):
    def __init__(self, num_channels=8):
        super().__init__()
        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        self.policy_conv = nn.Conv2d(128, 32, kernel_size=1)
        self.policy_bn = nn.BatchNorm2d(32)
        self.policy_fc = nn.Linear(32 * 9 * 9, 209)
        
        self.value_conv = nn.Conv2d(128, 1, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(9 * 9, 64)
        self.value_fc2 = nn.Linear(64, 1)

    def forward(self, x, action_mask=None):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        
        p = F.relu(self.policy_bn(self.policy_conv(x)))
        p = p.view(-1, 32 * 9 * 9)
        logits = self.policy_fc(p)
        
        # Masking illegal moves is correct
        if action_mask is not None:
            logits = logits.masked_fill(~action_mask, -1e9)
            
        v = F.relu(self.value_bn(self.value_conv(x)))
        v = v.view(-1, 9 * 9)
        v = F.relu(self.value_fc1(v))
        value = torch.tanh(self.value_fc2(v))
        
        return logits, value