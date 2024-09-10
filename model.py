import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm3d(out_channels)

        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm3d(out_channels)
            )

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class HybridSegResNetUNet(nn.Module):
    def __init__(self, in_channels, out_channels, init_filters=16, dropout_prob=0.2):
        super(HybridSegResNetUNet, self).__init__()
        
        self.encoder1 = ResidualBlock(in_channels, init_filters)
        self.encoder2 = ResidualBlock(init_filters, init_filters * 2, stride=2)
        self.encoder3 = ResidualBlock(init_filters * 2, init_filters * 4, stride=2)
        self.encoder4 = ResidualBlock(init_filters * 4, init_filters * 8, stride=2)
        
        self.bottleneck = ResidualBlock(init_filters * 8, init_filters * 16, stride=2)
        
        self.decoder4 = nn.ConvTranspose3d(init_filters * 16, init_filters * 8, kernel_size=2, stride=2)
        self.decoder3 = nn.ConvTranspose3d(init_filters * 8, init_filters * 4, kernel_size=2, stride=2)
        self.decoder2 = nn.ConvTranspose3d(init_filters * 4, init_filters * 2, kernel_size=2, stride=2)
        self.decoder1 = nn.ConvTranspose3d(init_filters * 2, init_filters, kernel_size=2, stride=2)
        
        self.final_conv = nn.Conv3d(init_filters, out_channels, kernel_size=1)
        
    def forward(self, x):
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)
        
        b = self.bottleneck(e4)
        
        d4 = self.decoder4(b)
        d4 = d4 + e4
        d4 = F.relu(d4)
        
        d3 = self.decoder3(d4)
        d3 = d3 + e3
        d3 = F.relu(d3)
        
        d2 = self.decoder2(d3)
        d2 = d2 + e2
        d2 = F.relu(d2)
        
        d1 = self.decoder1(d2)
        d1 = d1 + e1
        d1 = F.relu(d1)
        
        out = self.final_conv(d1)
        return out

device = torch.device("cuda:0")
model = HybridSegResNetUNet(
    in_channels=5, 
    out_channels=3, 
    init_filters=16, 
    dropout_prob=0.2
).to(device)

criterion = DiceLoss(sigmoid=True)  # Define or import DiceLoss
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
loss_function = DiceLoss(smooth_nr=0, smooth_dr=1e-5, squared_pred=True, to_onehot_y=False, sigmoid=True)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)
dice_metric = DiceMetric(include_background=True, reduction="mean")
dice_metric_batch = DiceMetric(include_background=True, reduction="mean_batch")
post_trans = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])

# define inference method
def inference(input):
    def _compute(input):
        return sliding_window_inference(
            inputs=input,
            roi_size=(240, 240, 160),
            sw_batch_size=1,
            predictor=model,
            overlap=0.5,
        )

    if VAL_AMP:
        with torch.cuda.amp.autocast():
            return _compute(input)
    else:
        return _compute(input)

    