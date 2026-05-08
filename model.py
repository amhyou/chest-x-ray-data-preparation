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
             ↓  [B, 128, 96, 96]          (2 max-pools → 384/4 = 96)
        Bridge Conv  (1×1, projects 128→128, learned)
             ↓  permute → [B, 96, 96, 128]
        Swin-Base  All 4 Stages  (pretrained, layers[0]–[3])
             ↓  [B, 12, 12, 1024]
        LayerNorm  +  GlobalAvgPool
             ↓  [B, 1024]
        SE Block
             ↓  [B, 1024]
        MLP Head  →  [B, num_classes]

    VGG16 Blocks 1-2 replace Swin's patch embedding — both produce
    a 96×96 feature map with 128 channels from a 384×384 input,
    so the shapes are compatible and Swin can run at full capacity.
    """

    def __init__(self, num_classes=4):
        super().__init__()

        # ── VGG16-BN Blocks 1–2 ──────────────────────────────────────────
        # features[0:14]:
        #   Block 1: Conv64×2 + BN + ReLU + MaxPool → [B, 64,  192, 192]
        #   Block 2: Conv128×2 + BN + ReLU + MaxPool → [B, 128,  96,  96]
        vgg = models.vgg16_bn(weights='IMAGENET1K_V1')
        self.backbone = vgg.features[:14]   # blocks 1–2 only

        # ── Bridge ───────────────────────────────────────────────────────
        # 1×1 conv to allow the model to adapt VGG's feature distribution
        # to Swin's expected input statistics. Output: [B, 128, 96, 96]
        swin_embed_dim = 128   # embed_dim of swin_base_patch4_window12_384
        self.bridge = nn.Sequential(
            nn.Conv2d(128, swin_embed_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(swin_embed_dim),
            nn.GELU()
        )

        # ── Full Swin-Base (pretrained, skip only patch_embed) ────────────
        # We bypass patch_embed (which would re-process the raw image) and
        # feed our VGG features directly into the transformer stages.
        swin = timm.create_model(
            'swin_base_patch4_window12_384',
            pretrained=True,
            num_classes=0,          # no classification head
            drop_path_rate=0.2
        )
        self.swin_layers  = swin.layers      # all 4 stages as ModuleList
        self.swin_norm    = swin.norm        # final LayerNorm
        self.embed_dim    = swin.num_features  # 1024 for Swin-Base

        # Aliases kept for C_train.py freeze-phase compatibility
        self.swin_model   = swin
        self.swin_stage2  = swin.layers[2]
        self.swin_stage3  = swin.layers[3]

        # ── SE Block + Classification Head ───────────────────────────────
        self.se = SEBlock(self.embed_dim)
        self.head = nn.Sequential(
            nn.Linear(self.embed_dim, 512),
            nn.GELU(),
            nn.Dropout(p=0.5),
            nn.Linear(512, num_classes)
        )

    def forward_features(self, x):
        # 1. VGG16 Blocks 1–2: RGB conv features
        x = self.backbone(x)                       # [B, 128, 96, 96]

        # 2. Bridge: adapt channel statistics
        x = self.bridge(x)                         # [B, 128, 96, 96]

        # 3. Reshape to [B, H, W, C] — timm Swin uses spatial 4D format
        x = x.permute(0, 2, 3, 1).contiguous()    # [B, 96, 96, 128]

        # 4. All Swin stages (replaces patch_embed + all 4 stages)
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