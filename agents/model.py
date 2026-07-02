import torch
import torch.nn as nn
import torch.nn.functional as F

class WallzNet(nn.Module):
    def __init__(self, num_channels=8): # Make sure this matches your new state channels!
        super().__init__()
        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        
        self.policy_conv = nn.Conv2d(128, 32, kernel_size=1)
        self.policy_fc = nn.Linear(32 * 9 * 9, 209)
        
        self.value_conv = nn.Conv2d(128, 1, kernel_size=1)
        self.value_fc1 = nn.Linear(9 * 9, 64)
        self.value_fc2 = nn.Linear(64, 1)

    def forward(self, x, action_mask=None):
        # 🔥 CHANGED: Using LeakyReLU(0.1) so the -1 values do not turn into 0
        x = F.leaky_relu(self.conv1(x), 0.1)
        x = F.leaky_relu(self.conv2(x), 0.1)
        x = F.leaky_relu(self.conv3(x), 0.1)
        
        p = F.leaky_relu(self.policy_conv(x), 0.1)
        p = p.view(-1, 32 * 9 * 9)
        logits = self.policy_fc(p)
        
        if action_mask is not None:
            logits = logits.masked_fill(~action_mask, -1e9)
            
        v = F.leaky_relu(self.value_conv(x), 0.1)
        v = v.view(-1, 9 * 9)
        v = F.leaky_relu(self.value_fc1(v), 0.1)
        value = torch.tanh(self.value_fc2(v))
        
        return logits, value
