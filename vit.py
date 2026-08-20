import torch
import torch.nn as nn

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=3,patch_size=16,embedding_dim=768):
        super().__init__()
        self.patch_size=patch_size
        self.projection=nn.Conv2d(
            in_channels,
            embedding_dim,
            kernel_size=patch_size,
            stride=patch_size
        )
        self.flatten=nn.Flatten(start_dim=2,end_dim=3)

    def forward(self,x):
        x=self.projection(x)
        x=self.flatten(x)
        x=x.permute(0,2,1)
        return x
class MultiHeadSelfAttention(nn.Module):
    def __init__(self,embedding_dim=768,num_heads=12,dropout=0.0):
        super().__init__()
        self.layernorm=nn.LayerNorm(normalized_shape=embedding_dim)
        self.attention=nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
    def forward(self,x):
        x=self.layernorm(x)
        out,_=self.attention(x,x,x)
        return out
class TransformerEncoderBlock(nn.Module):
    def __init__(self, embedding_dim=768,num_heads=12,mlp_size=3072,dropout=0.0):
        super().__init__()
        self.norm1=nn.LayerNorm(embedding_dim)
        self.attention_block=MultiHeadSelfAttention(embedding_dim,num_heads,dropout)
        self.norm2=nn.LayerNorm(embedding_dim)
        self.mlp_block=nn.Sequential(
            nn.LayerNorm(embedding_dim),
            nn.Linear(embedding_dim, mlp_size),  # expand 4x
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_size, embedding_dim),  # compress back
            nn.Dropout(dropout)
        )
    def forward(self,x):
        x=x+ self.attention_block(x)
        x=x+ self.mlp_block(x)
        return x
if __name__=="__main__":
    img=torch.randn(1,3,224,224)
    patch_embed=PatchEmbedding()
    out=patch_embed(img)
    print(f"After PatchEmbedding: {out.shape}")
    attn = MultiHeadSelfAttention()
    out = attn(out)
    print(f"After Attention: {out.shape}") 
        # test TransformerEncoderBlock
    block = TransformerEncoderBlock()
    out = block(out)
    print(f"After TransformerBlock: {out.shape}") 

class ViT(nn.Module):
    def __init__(self,img_size=224,patch_size=16,in_channels=3,num_classes=3,embedding_dim=768,num_heads=12,num_layers=12,mlp_dropout=0.1,attn_dropout=0.0):
        super().__init__()
        self.patch_embedding=PatchEmbedding(in_channels,patch_size,embedding_dim)
        num_patches=(img_size//patch_size)**2
        self.cls_token=nn.Parameter(torch.randn(1,1,embedding_dim))
        self.pos_embedding=nn.Parameter(torch.randn(1,num_patches+1,embedding_dim))
        self.transformer=nn.Sequential(
            *[TransformerEncoderBlock(embedding_dim,num_heads,mlp_dropout,attn_dropout)
              for i in range(num_layers)]
        )
        self.mlp_head=nn.Sequential(
            nn.LayerNorm(embedding_dim),
            nn.Linear(embedding_dim,num_classes)
        )
    def forward(self,x):
        B=x.shape[0]
        x=self.patch_embedding(x)
        cls_tokens=self.cls_token.expand(B,-1,-1)
        x=torch.cat([cls_tokens,x],dim=1)
        x=x+self.pos_embedding
        x=self.transformer(x)
        cls_output=x[:,0]
        return self.mlp_head(cls_output)
