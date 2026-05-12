import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import timm


class SEBlock(nn.Module):
    """Squeeze-and-Excitation block for channel attention."""
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.fc(x)


class VGGSwinHybridNet(nn.Module):
    """
    Serial Hybrid Architecture:

        Input (3 × 384 × 384)
             ↓
        VGG16-BN  Blocks 1–2  (pretrained, features[0:14])
             ↓  [B, 128, 96, 96]
        Bridge Conv 1×1  (128 → 96)  +  AdaptiveAvgPool2d(56×56)
             ↓  permute → [B, 56, 56, 96]
        Swin-Tiny  (swin_tiny_patch4_window7_224, pretrained, all 4 stages)
          Stage 0: [B, 28, 28, 192]
          Stage 1: [B, 14, 14, 384]
          Stage 2: [B,  7,  7, 768]
          Stage 3: [B,  7,  7, 768]
             ↓  LayerNorm + GlobalAvgPool
             ↓  [B, 768]
        SE Block  →  MLP Head  →  [B, num_classes]

    Swin Base is larger and more powerful.
    VGG handles the full 384×384 local feature extraction;
    the bridge maintains the 96×96 token grid that Swin Base
    (patch_size=4) expects for 384x384 inputs.
    """

    def __init__(self, num_classes=4, swin_model_name='swin_base_patch4_window12_384', drop_path_rate=0.2, head_dropout=0.5):
        super().__init__()

        # ── VGG16-BN Blocks 1–2 ──────────────────────────────────────────
        # features[0:14] → [B, 128, 96, 96] for 384×384 input
        vgg = models.vgg16_bn(weights='IMAGENET1K_V1')
        self.backbone = vgg.features[:14]   # blocks 1-2

        # ── Bridge ───────────────────────────────────────────────────────
        # Project to Swin's embed_dim (128 for base, 96 for tiny)
        swin_embed_dim = 96 if 'tiny' in swin_model_name else 128
        self.bridge = nn.Sequential(
            nn.Conv2d(128, swin_embed_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(swin_embed_dim),
            nn.GELU()
            # No pooling needed: 384x384 input -> VGG Blocks 1-2 -> 96x96 output
            # This perfectly matches Swin Base's native token grid for 384x384.
        )

        # ── Swin-Base (pretrained) ────────────────────────────────────────
        swin = timm.create_model(
            swin_model_name,
            pretrained=True,
            num_classes=0,
            drop_path_rate=drop_path_rate
        )
        self.swin_layers = swin.layers      # 4 stages as ModuleList
        self.swin_norm   = swin.norm        # final LayerNorm
        self.embed_dim   = swin.num_features  # 1024 for Swin-Base

        # Aliases for C_train.py freeze-phase compatibility
        self.swin_model  = swin
        self.swin_stage2 = swin.layers[2]
        self.swin_stage3 = swin.layers[3]

        # ── SE Block + Classification Head ───────────────────────────────
        self.se = SEBlock(self.embed_dim)
        self.head = nn.Sequential(
            nn.Linear(self.embed_dim, 512),
            nn.GELU(),
            nn.Dropout(p=head_dropout),
            nn.Linear(512, num_classes)
        )

    def forward_features(self, x):
        # 1. VGG16 Blocks 1-2
        x = self.backbone(x)                       # [B, 128, 96, 96]

        # 2. Bridge
        x = self.bridge(x)                         # [B, 128, 96, 96]

        # 3. [B, C, H, W] → [B, H, W, C]  (timm Swin 4D spatial format)
        x = x.permute(0, 2, 3, 1).contiguous()    # [B, 96, 96, 128]

        # 4. All Swin Base stages
        for layer in self.swin_layers:
            x = layer(x)
        # After all stages: [B, 12, 12, 1024]

        # 5. Norm + Global Average Pool
        x = self.swin_norm(x)                      # [B, 12, 12, 1024]
        x = x.mean(dim=(1, 2))                     # [B, 1024]
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.se(x)
        return self.head(x)