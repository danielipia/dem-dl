import torch.nn as nn
import torch

class DemoModel(nn.Module):
    def __init__(self):
        super(DemoModel, self).__init__()
        self.name = "DemoModel"
        self.net = nn.Sequential(
            nn.Conv2d(6, 512, kernel_size=1),
            nn.SiLU(),
            nn.Conv2d(512, 512, kernel_size=1),
            nn.SiLU(),
            nn.Conv2d(512, 256, kernel_size=1),
            nn.SiLU(),
            nn.Conv2d(256, 128, kernel_size=1),
            nn.SiLU(),
            nn.Conv2d(128, 64, kernel_size=1),
            nn.SiLU(),
            nn.Conv2d(64, 32, kernel_size=1),
            nn.SiLU(),
            nn.Conv2d(32, 18, kernel_size=1),
        )

    def forward(self, x):
        return self.net(x)

class DumbLinearModel(nn.Module):
    def __init__(self):
        super(DumbLinearModel, self).__init__()
        self.name = "DumbLinearModel"
        self.net = nn.Sequential(
            nn.Conv2d(6, 512, kernel_size=1),
            nn.Conv2d(512, 18, kernel_size=1),
        )

    def forward(self, x):
        return self.net(x)
    
class DEMBasisNet(nn.Module):
    def __init__(self, in_channels=6, out_channels=54):
        super().__init__()
        self.name = "DEMBasisNet"
        self.requires_basis = True
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 512, kernel_size=1),
            nn.SiLU(),
            nn.Conv2d(512, 512, kernel_size=1),
            nn.SiLU(),
            nn.Conv2d(512, 256, kernel_size=1),
            nn.SiLU(),
            nn.Conv2d(256, 128, kernel_size=1),
            nn.SiLU(),
            nn.Conv2d(128, out_channels, kernel_size=1),
            )

    def forward(self, x):  # x: [B, C, H, W]
        return self.net(x)
    
class BasisNetReLU(nn.Module):
    def __init__(self, in_channels=6, out_channels=54):
        super().__init__()
        self.name = "BasisNetReLU"
        self.requires_basis = True
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 512, kernel_size=1),
            nn.SiLU(),
            nn.Conv2d(512, 512, kernel_size=1),
            nn.SiLU(),
            nn.Conv2d(512, 256, kernel_size=1),
            nn.SiLU(),
            nn.Conv2d(256, 128, kernel_size=1),
            nn.SiLU(),
            nn.Conv2d(128, out_channels, kernel_size=1),
            nn.ReLU(),
            )

    def forward(self, x):  # x: [B, C, H, W]
        return self.net(x)

# maybe 2x?
# MLP Mixer
    
class BasisNetReLUBig(nn.Module):
    # 10× width multiplier on hidden channels
    def __init__(self, in_channels=6, out_channels=54):
        super().__init__()
        self.name = "BasisNetReLUBig"
        self.requires_basis = True
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 2048, kernel_size=1),  # 4×512
            nn.SiLU(),
            nn.Conv2d(2048,    2048, kernel_size=1),
            nn.SiLU(),
            nn.Conv2d(2048,    1024, kernel_size=1),  # 4×256
            nn.SiLU(),
            nn.Conv2d(1024,     512, kernel_size=1),  # 4×128
            nn.SiLU(),
            nn.Conv2d( 512, out_channels, kernel_size=1),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


class BasicNetworkFreq(nn.Module):
    def __init__(self, nIn=6, nOut=26, nFreq=12):
        super(BasicNetworkFreq, self).__init__()
        self.name = "BasicNetworkFreq"
        self.nFreq = nFreq
        self.nIn = nIn 
        self.nOut = nOut
        self.requires_basis = False


        nChannels = self.nIn*(self.nFreq*2+1)
#        nChannels = self.nIn

        self.layers = nn.Sequential(
                nn.Conv2d(nChannels, 128, 1, padding=0, padding_mode='reflect'),
                nn.SiLU(),
                nn.Dropout(p=0.2),
                nn.Conv2d(128, 256, 1, padding=0, padding_mode='reflect'),
                nn.SiLU(),
                nn.Dropout(p=0.2),
                nn.Conv2d(256, 256, 1, padding=0, padding_mode='reflect'),
                nn.SiLU(),
                nn.Dropout(p=0.2),
                nn.Conv2d(256, 256, 1, padding=0, padding_mode='reflect'),
                nn.SiLU(),
                nn.Dropout(p=0.2),
                nn.Conv2d(256, self.nOut, 1, padding=0),
                nn.ReLU())
        
    def forward(self, x):
        x = torch.pow(torch.clamp(x, 0, None), 0.5)
        x = posEncode(x, self.nFreq)
        return self.layers(x)

def posEncode(x, numFreq=16):
    # Map x in R^{F x W} -> R^{F*(2*num_freqs+1) x W} 
    # this is only so that dot products can represent high frequency detail
    # more easily
    F = x.shape[1]*(numFreq*2+1)
    posEnc = torch.zeros((x.shape[0], F, x.shape[2], x.shape[3]), device=x.device)
    # frequencies = 2*pi*2**i/2
    freqs = [6.28318530718*(2**(i/2.0)) for i in range(-4,numFreq-4)]
    for k in range(x.shape[1]):
        for j in range(numFreq):
            # don't blindly optimize this -- they need to be interleaved
            # in case we ever use multihead attention
            posEnc[:,(k*numFreq*2)+2*j] = torch.sin(freqs[j]*x[:,k])
            posEnc[:,(k*numFreq*2)+2*j+1] = torch.cos(freqs[j]*x[:,k])

    # add the feature at the end as a backup/safety
    posEnc[:,(x.shape[1]*numFreq*2):] = x
    return posEnc 



# basic freq class with 1x1 convs
class BasicNetworkFreqClass(nn.Module):
    def __init__(self, nIn=6, nOut=26, nFreq=12, n_bins=64):
        super().__init__()
        self.name = "BasicNetworkFreqClass"
        self.nFreq = nFreq
        self.nIn = nIn 
        self.nOut = nOut
        self.n_bins = n_bins
        self.requires_basis = False

        nChannels = self.nIn*(self.nFreq*2+1)
        self.backbone = nn.Sequential(
            nn.Conv2d(nChannels, 128, 1, padding=0, padding_mode='reflect'),
            nn.SiLU(),
            nn.Dropout(p=0.2),
            nn.Conv2d(128, 256, 1, padding=0, padding_mode='reflect'),
            nn.SiLU(),
            nn.Dropout(p=0.2),
            nn.Conv2d(256, 256, 1, padding=0, padding_mode='reflect'),
            nn.SiLU(),
            nn.Dropout(p=0.2),
            nn.Conv2d(256, 256, 1, padding=0, padding_mode='reflect'),
            nn.SiLU(),
            nn.Dropout(p=0.2),
        )
        self.regression_head = nn.Sequential(
            nn.Conv2d(256, self.nOut, 1, padding=0),
            nn.ReLU()
        )
        self.classification_head = nn.Conv2d(256, self.nOut * self.n_bins, 1, padding=0)

    def forward(self, x):
        x = torch.pow(torch.clamp(x, 0, None), 0.5)
        x = posEncode(x, self.nFreq)
        features = self.backbone(x)
        regression = self.regression_head(features)
        classification = self.classification_head(features)
        B, _, H, W = classification.shape
        classification = classification.view(B, self.nOut, self.n_bins, H, W)
        return regression, classification

# basic freq class with spatial convs
class BasicNetworkFreqClassConv(nn.Module):
    def __init__(self, nIn=6, nOut=26, nFreq=12, n_bins=64):
        super().__init__()
        self.name = "BasicNetworkFreqClassConv"
        self.nFreq = nFreq
        self.nIn = nIn
        self.nOut = nOut
        self.n_bins = n_bins
        self.requires_basis = False

        nChannels = self.nIn*(self.nFreq*2+1)
        self.backbone = nn.Sequential(
            nn.Conv2d(nChannels, 64, 3, padding=1, padding_mode='reflect'),
            nn.SiLU(),
            nn.Dropout(p=0.2),
            nn.Conv2d(64, 128, 3, padding=1, padding_mode='reflect'),
            nn.SiLU(),
            nn.Dropout(p=0.2),
            nn.Conv2d(128, 128, 3, padding=1, padding_mode='reflect'),
            nn.SiLU(),
            nn.Dropout(p=0.2),
            nn.Conv2d(128, 128, 3, padding=1, padding_mode='reflect'),
            nn.SiLU(),
            nn.Dropout(p=0.2),
        )
        self.regression_head = nn.Sequential(
            nn.Conv2d(128, self.nOut, 1, padding=0),
            nn.ReLU()
        )
        self.classification_head = nn.Conv2d(128, self.nOut * self.n_bins, 1, padding=0)

    def forward(self, x):
        x = torch.pow(torch.clamp(x, 0, None), 0.5)
        x = posEncode(x, self.nFreq)
        features = self.backbone(x)
        regression = self.regression_head(features)
        classification = self.classification_head(features)
        B, _, H, W = classification.shape
        classification = classification.view(B, self.nOut, self.n_bins, H, W)
        return regression, classification

# basic freq class, smaller
class BasicNetworkFreqClassSmall(nn.Module):
    def __init__(self, nIn=6, nOut=26, nFreq=12, n_bins=64):
        super().__init__()
        self.name = "BasicNetworkFreqClassSmall"
        self.nFreq = nFreq
        self.nIn = nIn
        self.nOut = nOut
        self.n_bins = n_bins
        self.requires_basis = False

        nChannels = self.nIn*(self.nFreq*2+1)
        self.backbone = nn.Sequential(
            nn.Conv2d(nChannels, 64, 1, padding=0, padding_mode='reflect'),
            nn.SiLU(),
            nn.Dropout(p=0.2),
            nn.Conv2d(64, 128, 1, padding=0, padding_mode='reflect'),
            nn.SiLU(),
            nn.Dropout(p=0.2),
            nn.Conv2d(128, 128, 1, padding=0, padding_mode='reflect'),
            nn.SiLU(),
            nn.Dropout(p=0.2),
            nn.Conv2d(128, 128, 1, padding=0, padding_mode='reflect'),
            nn.SiLU(),
            nn.Dropout(p=0.2),
        )
        self.regression_head = nn.Sequential(
            nn.Conv2d(128, self.nOut, 1, padding=0),
            nn.ReLU()
        )
        self.classification_head = nn.Conv2d(128, self.nOut * self.n_bins, 1, padding=0)

    def forward(self, x):
        x = torch.pow(torch.clamp(x, 0, None), 0.5)
        x = posEncode(x, self.nFreq)
        features = self.backbone(x)
        regression = self.regression_head(features)
        classification = self.classification_head(features)
        B, _, H, W = classification.shape
        classification = classification.view(B, self.nOut, self.n_bins, H, W)
        return regression, classification

##### EXPERIMENTAL #####

import torch
import torch.nn as nn
import torch.nn.functional as F

# simple per-pixel mixer block: mlp over channels, no norm, rezero
class ChannelMLPBlockNoNorm(nn.Module):
    def __init__(self, c_in, c_hidden, p=0.2, zero_last=True):
        super().__init__()
        # 1x1 convs == per-pixel linear over channels
        self.fc1 = nn.Conv2d(c_in, c_hidden, kernel_size=1, bias=True)
        self.fc2 = nn.Conv2d(c_hidden, c_in, kernel_size=1, bias=True)
        self.drop = nn.Dropout(p)
        # rezero scale
        self.alpha = nn.Parameter(torch.zeros(1))

        # init (stable, no norm)
        nn.init.kaiming_normal_(self.fc1.weight, nonlinearity='relu')
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)
        nn.init.kaiming_normal_(self.fc2.weight, nonlinearity='linear')

    def forward(self, x):
        y = self.fc1(x)
        y = F.gelu(y)
        y = self.drop(y)
        y = self.fc2(y)
        return x + self.alpha * y

# optional: tiny spatial mixing that won’t break stability (disabled by default)
class DepthwiseTokenMix(nn.Module):
    def __init__(self, c, kernel_size=3, p=0.0):
        super().__init__()
        pad = kernel_size // 2
        self.dw = nn.Conv2d(c, c, kernel_size, padding=pad, groups=c, bias=True)
        self.drop = nn.Dropout(p)
        self.alpha = nn.Parameter(torch.zeros(1))

        nn.init.kaiming_normal_(self.dw.weight, nonlinearity='linear')
        nn.init.zeros_(self.dw.bias)

    def forward(self, x):
        y = self.dw(x)
        y = self.drop(F.gelu(y))
        return x + self.alpha * y

class NoNormMixer(nn.Module):
    def __init__(self, nIn=6, nOut=26, nFreq=12, n_bins=64,
                 width=128, hidden_ratio=1.0, n_blocks=6,
                 p_dropout=0.2, use_token_mix=True):
        super().__init__()
        self.name = "NoNormMixer"
        self.nFreq = nFreq
        self.nIn = nIn
        self.nOut = nOut
        self.n_bins = n_bins
        self.requires_basis = False

        nChannels = self.nIn * (self.nFreq * 2 + 1)

        # simple 1x1 lift to model width
        self.stem = nn.Conv2d(nChannels, width, kernel_size=1, padding=0, bias=True)
        nn.init.kaiming_normal_(self.stem.weight, nonlinearity='linear')
        nn.init.zeros_(self.stem.bias)

        c_hidden = int(width * hidden_ratio)

        blocks = []
        for _ in range(n_blocks):
            if use_token_mix:
                blocks.append(DepthwiseTokenMix(width, kernel_size=3, p=0.0))
            blocks.append(ChannelMLPBlockNoNorm(width, c_hidden, p=p_dropout, zero_last=True))
        self.backbone = nn.Sequential(*blocks)

        # heads (unchanged API)
        self.regression_head = nn.Sequential(
            nn.Conv2d(width, self.nOut, 1, padding=0, bias=True),
            nn.ReLU()

            # maybe take out relu here?
        )
        self.classification_head = nn.Conv2d(width, self.nOut * self.n_bins, 1, padding=0, bias=True)

        # final-layer init
        nn.init.zeros_(self.regression_head[0].bias)
        nn.init.kaiming_normal_(self.regression_head[0].weight, nonlinearity='relu')
        nn.init.zeros_(self.classification_head.bias)
        nn.init.kaiming_normal_(self.classification_head.weight, nonlinearity='linear')

    def forward(self, x):
        # your preprocessing + fourier features
        x = torch.pow(torch.clamp(x, 0, None), 0.5)
        x = posEncode(x, self.nFreq)

        x = self.stem(x)
        features = self.backbone(x)

        regression = self.regression_head(features)
        classification = self.classification_head(features)
        B, _, H, W = classification.shape
        classification = classification.view(B, self.nOut, self.n_bins, H, W)
        return regression, classification


# basic freq class with 1x1 convs
class FreqClassNonReLU(nn.Module):
    def __init__(self, nIn=6, nOut=26, nFreq=12, n_bins=64):
        super().__init__()
        self.name = "FreqClassNonReLU"
        self.nFreq = nFreq
        self.nIn = nIn 
        self.nOut = nOut
        self.n_bins = n_bins
        self.requires_basis = False

        nChannels = self.nIn*(self.nFreq*2+1)
        self.backbone = nn.Sequential(
            nn.Conv2d(nChannels, 128, 1, padding=0, padding_mode='reflect'),
            nn.SiLU(),
            nn.Dropout(p=0.2),
            nn.Conv2d(128, 256, 1, padding=0, padding_mode='reflect'),
            nn.SiLU(),
            nn.Dropout(p=0.2),
            nn.Conv2d(256, 256, 1, padding=0, padding_mode='reflect'),
            nn.SiLU(),
            nn.Dropout(p=0.2),
            nn.Conv2d(256, 256, 1, padding=0, padding_mode='reflect'),
            nn.SiLU(),
            nn.Dropout(p=0.2),
        )
        self.regression_head = nn.Sequential(
            nn.Conv2d(256, self.nOut, 1, padding=0)
        )
        self.classification_head = nn.Conv2d(256, self.nOut * self.n_bins, 1, padding=0)

    def forward(self, x):
        x = torch.pow(torch.clamp(x, 0, None), 0.5)
        x = posEncode(x, self.nFreq)
        features = self.backbone(x)
        regression = self.regression_head(features)
        classification = self.classification_head(features)
        B, _, H, W = classification.shape
        classification = classification.view(B, self.nOut, self.n_bins, H, W)
        return regression, classification


class FreqClassNonReLUNoPos(nn.Module):
    """nonrelu variant without positional encoding"""
    def __init__(self, nIn=6, nOut=26, nFreq=12, n_bins=64):
        super().__init__()
        self.name = "FreqClassNonReLUNoPos"
        self.nFreq = 0  # no positional encoding
        self.nIn = nIn 
        self.nOut = nOut
        self.n_bins = n_bins
        self.requires_basis = False

        # no pos encoding, so just nIn channels
        nChannels = self.nIn
        self.backbone = nn.Sequential(
            nn.Conv2d(nChannels, 128, 1, padding=0, padding_mode='reflect'),
            nn.SiLU(),
            nn.Dropout(p=0.2),
            nn.Conv2d(128, 256, 1, padding=0, padding_mode='reflect'),
            nn.SiLU(),
            nn.Dropout(p=0.2),
            nn.Conv2d(256, 256, 1, padding=0, padding_mode='reflect'),
            nn.SiLU(),
            nn.Dropout(p=0.2),
            nn.Conv2d(256, 256, 1, padding=0, padding_mode='reflect'),
            nn.SiLU(),
            nn.Dropout(p=0.2),
        )
        self.regression_head = nn.Sequential(
            nn.Conv2d(256, self.nOut, 1, padding=0)
        )
        self.classification_head = nn.Conv2d(256, self.nOut * self.n_bins, 1, padding=0)

    def forward(self, x):
        x = torch.pow(torch.clamp(x, 0, None), 0.5)
        # skip positional encoding
        features = self.backbone(x)
        regression = self.regression_head(features)
        classification = self.classification_head(features)
        B, _, H, W = classification.shape
        classification = classification.view(B, self.nOut, self.n_bins, H, W)
        return regression, classification
