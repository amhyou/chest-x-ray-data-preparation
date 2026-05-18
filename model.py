import torch
import torch.nn as nn
import torchvision.models as models
import timm
import config

class VGGSwinHybridNet(nn.Module):
    """
    Lightweight Serial Hybrid Architecture:
        Input (3 × IMG_SIZE × IMG_SIZE)
             ↓
        VGG11-BN  Blocks 1–2  (pretrained, features[0:8]) -> ONLY 2 CONV LAYERS
             ↓  [B, 128, H/4, W/4]
        Bridge Conv 1×1  (128 → 96)
             ↓  permute → [B, H/4, W/4, 96]
        Swin-Tiny  (pretrained, all 4 stages)
             ↓  LayerNorm + GlobalAvgPool
        Linear Classifier  →  [B, num_classes]
    """

    def __init__(self, num_classes=4, swin_model_name=None, drop_path_rate=0.2, head_dropout=0.2):
        super().__init__()

        # Force lightweight Swin model
        if swin_model_name is None:
            if config.IMG_SIZE == 224:
                swin_model_name = 'swin_tiny_patch4_window7_224'
            else:
                swin_model_name = 'swin_tiny_patch4_window12_384'

        # ── Lighter VGG11-BN Blocks 1–2 ───────────────────────────────────
        # VGG11 is vastly lighter than VGG16.
        # features[:8] corresponds to exactly the first 2 pooling stages:
        # Conv->BN->ReLU->MaxPool -> Conv->BN->ReLU->MaxPool. (Downsamples by 4x).
        vgg = models.vgg11_bn(weights='IMAGENET1K_V1')
        self.backbone = vgg.features[:8]

        # ── Bridge ───────────────────────────────────────────────────────
        swin_embed_dim = 96 # Swin-Tiny embed dim is 96
        self.bridge = nn.Sequential(
            nn.Conv2d(128, swin_embed_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(swin_embed_dim),
            nn.GELU()
        )

        # ── Swin-Tiny (pretrained) ────────────────────────────────────────
        swin = timm.create_model(
            swin_model_name,
            pretrained=True,
            num_classes=0,
            drop_path_rate=drop_path_rate
        )
        self.swin_layers = swin.layers      # 4 stages as ModuleList
        self.swin_norm   = swin.norm        # final LayerNorm
        self.embed_dim   = swin.num_features  # 768 for Swin-Tiny

        # Aliases for C_train.py freeze-phase compatibility and GradCAM
        self.swin_model  = swin
        self.swin_stage2 = swin.layers[2]
        self.swin_stage3 = swin.layers[3]

        # ── Minimal Classification Head ───────────────────────────────────
        # Removed heavy SE block and huge 512-dim MLP bottleneck to prevent overfitting.
        self.dropout = nn.Dropout(p=head_dropout)
        self.head = nn.Linear(self.embed_dim, num_classes)

    def forward_features(self, x):
        # 1. VGG11 Blocks 1-2
        x = self.backbone(x)                       # [B, 128, H/4, W/4]

        # 2. Bridge
        x = self.bridge(x)                         # [B, 96, H/4, W/4]

        # 3. [B, C, H, W] → [B, H, W, C]  (timm Swin 4D spatial format)
        x = x.permute(0, 2, 3, 1).contiguous()    # [B, H/4, W/4, 96]

        # 4. All Swin stages
        for layer in self.swin_layers:
            x = layer(x)

        # 5. Norm + Global Average Pool
        x = self.swin_norm(x)
        x = x.mean(dim=(1, 2))                     # [B, 768]
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.dropout(x)
        return self.head(x)