# file: model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import math

# =================================================================================
# HELPER MODULES (Unchanged)
# =================================================================================

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class Block(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.time_mlp = nn.Linear(time_emb_dim, out_ch)

    def forward(self, x, t):
        h = self.relu(self.conv1(x))
        time_emb = self.relu(self.time_mlp(t))
        time_emb = time_emb.unsqueeze(-1).unsqueeze(-1)
        h = h + time_emb
        h = self.relu(self.conv2(h))
        return h

# =================================================================================
# NEW: VGG-FPN CONDITIONER FOR 224x224 INPUT
# =================================================================================

class VGG16_FPN_224x224(nn.Module):
    """
    VGG16-FPN adapted for a 224x224 input.
    Outputs a feature map of shape (B, out_channels, 56, 56), which corresponds
    to the P2 level for a 224x224 input.
    """
    def __init__(self, out_channels=128, pretrained=True):
        super(VGG16_FPN_224x224, self).__init__()
        
        self.out_channels = out_channels
        
        vgg16 = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1 if pretrained else None)
        self.features = vgg16.features
        
        # --- MODIFICATION: Redefined stages for 224x224 input ---
        # Tracing spatial dimensions: 224 -> 112 -> 56 -> 28 -> 14 -> 7
        self.stage2 = nn.Sequential(*self.features[:10])   # Output C2: 128 channels, 56x56
        self.stage3 = nn.Sequential(*self.features[10:17]) # Output C3: 256 channels, 28x28
        self.stage4 = nn.Sequential(*self.features[17:24]) # Output C4: 512 channels, 14x14
        self.stage5 = nn.Sequential(*self.features[24:31]) # Output C5: 512 channels, 7x7

        self.lateral_c2 = nn.Conv2d(128, self.out_channels, kernel_size=1)
        self.lateral_c3 = nn.Conv2d(256, self.out_channels, kernel_size=1)
        self.lateral_c4 = nn.Conv2d(512, self.out_channels, kernel_size=1)
        self.lateral_c5 = nn.Conv2d(512, self.out_channels, kernel_size=1)
        
        self.smooth_p2 = nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, padding=1)
        self.smooth_p3 = nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, padding=1)
        self.smooth_p4 = nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, padding=1)

    def _upsample_add(self, x, y):
        _, _, H, W = y.size()
        return F.interpolate(x, size=(H, W), mode='bilinear', align_corners=False) + y

    def forward(self, x):
        c2 = self.stage2(x)
        c3 = self.stage3(c2)
        c4 = self.stage4(c3)
        c5 = self.stage5(c4)
        
        p5 = self.lateral_c5(c5)
        p4 = self._upsample_add(p5, self.lateral_c4(c4))
        p4 = self.smooth_p4(p4)
        p3 = self._upsample_add(p4, self.lateral_c3(c3))
        p3 = self.smooth_p3(p3)
        p2 = self._upsample_add(p3, self.lateral_c2(c2))
        p2 = self.smooth_p2(p2) # Shape: (B, out_channels, 56, 56)
        
        return p2

# =================================================================================
# NEW: UNET FOR 224x224 INPUT
# =================================================================================

class UNet224(nn.Module):
    """
    A U-Net with additional down/up sampling levels to handle 224x224 images.
    Conditioning happens at the 56x56 feature map level.
    """
    def __init__(self, in_channels=1, time_emb_dim=32, num_classes=2):
        super().__init__()
        
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.ReLU(),
        )
        
        # --- MODIFICATION: Deeper U-Net architecture for 224x224 ---
        # Downsampling path
        self.down1 = Block(in_channels, 64, time_emb_dim)
        self.pool1 = nn.MaxPool2d(2) # 224 -> 112
        self.down2 = Block(64, 128, time_emb_dim)
        self.pool2 = nn.MaxPool2d(2) # 112 -> 56
        self.down3 = Block(128, 256, time_emb_dim)
        self.pool3 = nn.MaxPool2d(2) # 56 -> 28
        self.down4 = Block(256, 512, time_emb_dim)
        self.pool4 = nn.MaxPool2d(2) # 28 -> 14

        # Bottleneck
        self.bot1 = Block(512, 1024, time_emb_dim)

        # Upsampling path
        self.upconv1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2) # 14 -> 28
        self.up1 = Block(512 + 512, 512, time_emb_dim) # Cat with down4
        self.upconv2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2) # 28 -> 56
        self.up2 = Block(256 + 256, 256, time_emb_dim) # Cat with down3
        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2) # 56 -> 112
        self.up3 = Block(128 + 128, 128, time_emb_dim) # Cat with down2
        self.upconv4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2) # 112 -> 224
        self.up4 = Block(64 + 64, 64, time_emb_dim) # Cat with down1
        
        self.out = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x, t, condition_features=None):
        x = x * 2 - 1
        t_emb = self.time_mlp(t)
        
        # Down path
        x1 = self.down1(x, t_emb)    # 224x224
        p1 = self.pool1(x1)          # 112x112
        x2 = self.down2(p1, t_emb)   # 112x112
        p2 = self.pool2(x2)          # 56x56
        
        # --- MODIFICATION: Conditioning happens at the 56x56 level ---
        if condition_features is not None:
            # Assumes condition_features is (B, 128, 56, 56)
            p2 = p2 + condition_features
        
        x3 = self.down3(p2, t_emb)   # 56x56
        p3 = self.pool3(x3)          # 28x28
        x4 = self.down4(p3, t_emb)   # 28x28
        p4 = self.pool4(x4)          # 14x14

        # Bottleneck
        b = self.bot1(p4, t_emb)
        
        # Up path
        u1 = self.upconv1(b)
        u1 = torch.cat([u1, x4], dim=1)
        u1 = self.up1(u1, t_emb)
        
        u2 = self.upconv2(u1)
        u2 = torch.cat([u2, x3], dim=1)
        u2 = self.up2(u2, t_emb)
        
        u3 = self.upconv3(u2)
        u3 = torch.cat([u3, x2], dim=1)
        u3 = self.up3(u3, t_emb)
        
        u4 = self.upconv4(u3)
        u4 = torch.cat([u4, x1], dim=1)
        u4 = self.up4(u4, t_emb)

        return self.out(u4)

# =================================================================================
# NEW: TOP-LEVEL CONDITIONAL MODEL FOR 224x224
# =================================================================================

class ConditionalDiffusionModel224(nn.Module):
    def __init__(self, unet_in_channels=1, time_emb_dim=32, num_classes=2, vgg_pretrained=True):
        super().__init__()
        
        # The U-Net is conditioned at 56x56 with 128 channels.
        # The FPN conditioner must match this.
        unet_condition_channels = 128
        
        self.conditioner = VGG16_FPN_224x224(
            out_channels=unet_condition_channels, 
            pretrained=vgg_pretrained
        )
        
        self.generator = UNet224(
            in_channels=unet_in_channels,
            time_emb_dim=time_emb_dim,
            num_classes=num_classes
        )

    def forward(self, noisy_map, t, crowd_image):
        condition_features = self.conditioner(crowd_image)
        logits = self.generator(noisy_map, t, condition_features)
        return logits

# =================================================================================
# DUMMY TESTS AND PARAMETER COUNTING FOR 224x224 MODEL
# =================================================================================

if __name__ == '__main__':
    print("--- Running ConditionalDiffusionModel224 Test ---")
    device = "cpu"
    
    # Instantiate the 224x224 model
    model = ConditionalDiffusionModel224(vgg_pretrained=False).to(device)
    
    # Create 224x224 dummy inputs
    batch_size = 2
    dummy_noisy_map = torch.rand(batch_size, 1, 224, 224, device=device)
    dummy_t = torch.randint(0, 200, (batch_size,), device=device).long()
    dummy_crowd_image = torch.randn(batch_size, 3, 224, 224, device=device)
    
    # Forward pass
    predicted_logits = model(dummy_noisy_map, dummy_t, dummy_crowd_image)
    
    print(f"Input shape (Noisy Map):  {dummy_noisy_map.shape}")
    print(f"Input shape (Crowd Img):  {dummy_crowd_image.shape}")
    print(f"Output shape (Logits):    {predicted_logits.shape}")
    
    expected_shape = (batch_size, 2, 224, 224)
    assert predicted_logits.shape == expected_shape, f"Shape mismatch! Expected {expected_shape}, got {predicted_logits.shape}"
    print("ConditionalDiffusionModel224 test passed successfully!")

    # --- Trainable Parameter Count Analysis for 224x224 Model ---
    print("\n--- Trainable Parameter Count Analysis (224x224 Model) ---")

    params_conditioner = sum(p.numel() for p in model.conditioner.parameters() if p.requires_grad)
    params_generator = sum(p.numel() for p in model.generator.parameters() if p.requires_grad)
    params_total = params_conditioner + params_generator

    print(f"Trainable parameters in VGG-FPN (224x224): {params_conditioner:,}")
    print(f"Trainable parameters in U-Net (224x224):   {params_generator:,}")
    print(f"---------------------------------------------")
    print(f"Total Trainable Parameters:                {params_total:,}")
