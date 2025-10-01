import torch
from torch import nn, Tensor
import math

class LoRALayer(nn.Module):
    def __init__(self, original_layer: nn.Linear, rank: int = 8, alpha: int = 16):
        super().__init__()
        self.original_layer = original_layer 
        
        in_features = original_layer.in_features
        out_features = original_layer.out_features
        
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        
        self.scaling = alpha / rank
        self.dropout = nn.Dropout(0.1)
        
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> Tensor:
        original_output = self.original_layer(x)
        
        lora_update = self.lora_B @ self.lora_A
        lora_output = self.dropout(x) @ lora_update.T * self.scaling
        
        return original_output + lora_output


class BidirectionalAttention(nn.Module):
    def __init__(self, k1_dim, k2_dim, v1_dim, v2_dim, attention_dim):
        super().__init__()
        self.k1_layer = nn.Linear(k1_dim, attention_dim)
        self.k2_layer = nn.Linear(k2_dim, attention_dim)
        self.softmax1 = nn.Softmax(dim=-1)
        self.softmax2 = nn.Softmax(dim=-1)

    def forward(self, k1, k2, v1, v2, k1_lengths=None, k2_lengths=None):
        k1_proj = self.k1_layer(k1)
        k2_proj = self.k2_layer(k2)
        
        score = torch.bmm(k1_proj, k2_proj.transpose(1, 2))
        
        w1 = self.softmax1(score.transpose(1, 2))
        w2 = self.softmax2(score)

        o1 = torch.bmm(w1, v1)
        o2 = torch.bmm(w2, v2)
        
        return o2, o1


class CoAttentionFusionBlock(nn.Module):
    def __init__(self, d_model: int, nhead: int):
        super().__init__()
        
        self.bidirectional_attention = BidirectionalAttention(
            k1_dim=d_model, k2_dim=d_model,
            v1_dim=d_model, v2_dim=d_model,
            attention_dim=d_model
        )
        
        self.final_proj = nn.Linear(d_model * 2, d_model)

    def forward(self, feat_rgb: Tensor, feat_depth: Tensor) -> Tensor:
        feat_rgb_seq = feat_rgb.unsqueeze(1)
        feat_depth_seq = feat_depth.unsqueeze(1)

        enhanced_rgb_seq, enhanced_depth_seq = self.bidirectional_attention(
            k1=feat_rgb_seq, k2=feat_depth_seq,
            v1=feat_rgb_seq, v2=feat_depth_seq
        )
        
        enhanced_rgb = enhanced_rgb_seq.squeeze(1)
        enhanced_depth = enhanced_depth_seq.squeeze(1)
        
        fused_input = torch.cat([enhanced_rgb, enhanced_depth], dim=1)
        fused_output = self.final_proj(fused_input)
        
        return fused_output
