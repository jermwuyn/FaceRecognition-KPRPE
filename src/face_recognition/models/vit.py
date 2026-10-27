import torch
import torch.nn as nn
from src.face_recognition.models.kprpe import KPRPE


class PatchEmbedding(nn.Module):
    def __init__(
        self,
        image_size=112,
        patch_size=8,
        in_channels=3,
        embed_dim=192,
    ):
        super().__init__()

        if image_size % patch_size != 0:
            raise ValueError(
                "image_size must be divisible by patch_size"
            )

        self.num_patches = (image_size // patch_size) ** 2

        self.projection = nn.Conv2d(
            in_channels=in_channels,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x,):
        x = self.projection(x)
        x = x.flatten(2)
        x = x.transpose(1, 2)

        return x

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim=192, num_heads=3, attn_dropout=0.0):
        super().__init__()

        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")

        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.attn_drop = nn.Dropout(attn_dropout)
        self.proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x, attention_bias = None):
        B, N, C = x.shape

        qkv = self.qkv(x)
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)

        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale

        if attention_bias is not None:
            expected_shape = (B, 1, N, N)

            if attention_bias.shape != expected_shape:
                raise ValueError(
                    f"Expected attention_bias shape {expected_shape},"
                    f"got {tuple(attention_bias.shape)}"
                )
            attn = attn + attention_bias

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = attn @ v
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)

        return x

class TransformerEncoderBlock(nn.Module):
    def __init__(
        self,
        embed_dim=192,
        num_heads=3,
        mlp_ratio=4.0,
        dropout=0.0,
    ):
        super().__init__()

        self.norm1 = nn.LayerNorm(embed_dim)

        self.attention = MultiHeadSelfAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            attn_dropout=dropout,
        )

        self.norm2 = nn.LayerNorm(embed_dim)

        hidden_dim = int(embed_dim * mlp_ratio)

        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x, attention_bias = None):
        x = x + self.attention(
            self.norm1(x),
            attention_bias = attention_bias)
        x = x + self.mlp(self.norm2(x))

        return x

class VisionTransformer(nn.Module):
    def __init__(
        self,
        image_size=112,
        patch_size=8,
        embed_dim=192,
        depth=6,
        num_heads=3,
        mlp_ratio=4.0,
    ):
        super().__init__()

        self.patch_embed = PatchEmbedding(
            image_size=image_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
        )

        self.kprpe = KPRPE(
            image_size = image_size,
            patch_size = patch_size,
        )    

        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.patch_embed.num_patches, embed_dim)
        )

        self.blocks = nn.ModuleList([
            TransformerEncoderBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
            )
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)

        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x, landmarks = None):
        x = self.patch_embed(x)
        x = x + self.pos_embed

        attention_bias = None

        if landmarks is not None:
            attention_bias = self.kprpe(landmarks)  

        for block in self.blocks:
            x = block(x, attention_bias = attention_bias)

        x = self.norm(x)

        return x

class FaceEmbeddingHead(nn.Module):
    def __init__(self, embed_dim=192, output_dim=256):
        super().__init__()

        self.norm = nn.LayerNorm(embed_dim)
        self.projection = nn.Linear(embed_dim, output_dim)

    def forward(self, tokens):
        # Input: [B, 196, 192]
        x = tokens.mean(dim=1)

        # [B, 192]
        x = self.norm(x)

        # [B, 256]
        x = self.projection(x)

        # L2-normalized face embeddings
        x = torch.nn.functional.normalize(x, p=2, dim=-1)

        return x 

class FaceRecognitionModel(nn.Module):
    def __init__(
        self,
        image_size=112,
        patch_size=8,
        embed_dim=192,
        depth=6,
        num_heads=3,
        mlp_ratio=4.0,
        embedding_dim=256,
    ):
        super().__init__()

        self.backbone = VisionTransformer(
            image_size=image_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
        )

        self.embedding_head = FaceEmbeddingHead(
            embed_dim=embed_dim,
            output_dim=embedding_dim,
        )

    def forward(self, images, landmarks=None):
        tokens = self.backbone(
            images,
            landmarks=landmarks,
        )

        embeddings = self.embedding_head(tokens)

        return embeddings