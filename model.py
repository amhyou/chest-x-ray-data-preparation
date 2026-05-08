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
        VGG16-BN  Blocks 1–4  (pretrained, features[0:34])
             ↓  [B, 512, 24, 24]
        Bridge Conv  (1×1, learned projection)
             ↓  flatten → [B, 576, 512]   (576 = 24 × 24)
        Swin-Base  Stage 2  (layers[2], 18 blocks)  — pretrained
             ↓  [B, 144, 1024]            (12 × 12 after PatchMerging)
        Swin-Base  Stage 3  (layers[3],  2 blocks)  — pretrained
             ↓  [B, 144, 1024]
        LayerNorm  +  GlobalAvgPool
             ↓  [B, 1024]
        SE Block
             ↓  [B, 1024]
        MLP Head  →  [B, num_classes]

    VGG16 Block 5 is intentionally skipped; Swin Stages 0–1 are
    replaced by VGG's spatial feature extraction, avoiding redundant
    patch-level downsampling.
    """

    def __init__(self, num_classes=5):
        super().__init__()

        # ── VGG16-BN Blocks 1–4 ──────────────────────────────────────────
        # features[0:34] covers blocks 1-4 ending with MaxPool at index 33.
        # Output for 384×384 input: [B, 512, 24, 24]
        vgg = models.vgg16_bn(weights='IMAGENET1K_V1')
        self.backbone = vgg.features[:34]   # blocks 1–4 only

        # ── Bridge: 1×1 Conv → project channels for Swin ─────────────────
        # Swin-Base Stage 2 expects [B, N, 512] where N = 24*24 = 576
        self.bridge = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=1, bias=False),
            nn.BatchNorm2d(512),
            nn.GELU()
        )

        # ── Swin-Base (pretrained), Stages 2 and 3 only ──────────────────
        # We skip patch_embed, layers[0], layers[1] — VGG already handled
        # the equivalent spatial downsampling (384→24 via 4 max-pools).
        swin = timm.create_model(
            'swin_base_patch4_window12_384',
            pretrained=True,
            num_classes=0,          # removes classification head
            drop_path_rate=0.2
        )
        # Keep only what we use
        self.swin_stage2 = swin.layers[2]   # 18 blocks, downsamples 24→12
        self.swin_stage3 = swin.layers[3]   # 2 blocks, no downsampling
        self.swin_norm   = swin.norm        # LayerNorm before pooling
        self.swin_pool   = nn.AdaptiveAvgPool1d(1)
        self.embed_dim   = swin.num_features   # 1024 for Swin-Base

        # Alias kept for C_train.py freeze compatibility
        self.swin_model = swin

        # ── SE Block + Classification Head ───────────────────────────────
        self.se = SEBlock(self.embed_dim)

        self.head = nn.Sequential(
            nn.Linear(self.embed_dim, 512),
            nn.GELU(),
            nn.Dropout(p=0.5),
            nn.Linear(512, num_classes)
        )

    def forward_features(self, x):
        # 1. VGG16 Blocks 1–4
        x = self.backbone(x)                     # [B, 512, 24, 24]

        # 2. Bridge
        x = self.bridge(x)                        # [B, 512, 24, 24]
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)          # [B, 576, 512]

        # 3. Swin Stages 2 and 3
        x = self.swin_stage2(x)                   # [B, 144, 1024]
        x = self.swin_stage3(x)                   # [B, 144, 1024]
        x = self.swin_norm(x)                     # [B, 144, 1024]

        # 4. Global Average Pool
        x = self.swin_pool(x.transpose(1, 2))     # [B, 1024, 1]
        x = x.flatten(1)                           # [B, 1024]
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.se(x)
        return self.head(x)