import torch
from torch import nn, Tensor
import torch.nn.functional as F

class NormalPrototypes(nn.Module):
    def __init__(self, num_prototypes: int, feature_dim: int):
        super().__init__()
        self.prototypes = nn.Parameter(torch.randn(num_prototypes, feature_dim))

    def forward(self, per_point_features: Tensor) -> Tensor:
        point_feats_norm = F.normalize(per_point_features, p=2, dim=-1)
        prototypes_norm = F.normalize(self.prototypes, p=2, dim=-1)
        similarity_matrix = torch.matmul(point_feats_norm, prototypes_norm.t())
        max_similarity, _ = torch.max(similarity_matrix, dim=2)
        outlier_scores = 1.0 - max_similarity
        return outlier_scores.unsqueeze(-1)

class OutlierFeatureAggregator(nn.Module):
    def __init__(self, num_defect_tokens: int, top_k: int, point_feat_dim: int, text_embed_dim: int, nhead: int):
        super().__init__()
        self.top_k = top_k
        self.num_defect_tokens = num_defect_tokens
        self.point_feat_proj = nn.Linear(point_feat_dim, text_embed_dim)
        self.aggregator_attention = nn.MultiheadAttention(text_embed_dim, nhead, batch_first=True)
        self.norm = nn.LayerNorm(text_embed_dim)
        self.ffn = nn.Sequential(nn.Linear(text_embed_dim, text_embed_dim), nn.GELU())

    def forward(self, per_point_features: Tensor, outlier_scores: Tensor) -> Tensor:
        B, N, D = per_point_features.shape
        k = min(self.top_k, N)
        _, top_k_indices = torch.topk(outlier_scores.squeeze(-1), k=k, dim=1)
        
        top_k_indices_expanded = top_k_indices.unsqueeze(-1).expand(-1, -1, D)
        outlier_feats = torch.gather(per_point_features, 1, top_k_indices_expanded)
        
        outlier_feats_proj = self.point_feat_proj(outlier_feats)
        attn_output, _ = self.aggregator_attention(outlier_feats_proj, outlier_feats_proj, outlier_feats_proj)
        aggregated_feat = self.norm(outlier_feats_proj + attn_output)
        aggregated_feat = aggregated_feat + self.ffn(aggregated_feat)
        
            
        summary_tokens = aggregated_feat[:, :self.num_defect_tokens, :]
        
        return summary_tokens

class DefectPromptGenerator(nn.Module):
    def __init__(self, 
                 point_feat_dim: int = 128, 
                 text_embed_dim: int = 768, 
                 num_defect_tokens: int = 4, 
                 num_normal_prototypes: int = 32, 
                 top_k_outliers: int = 32, 
                 nhead: int = 4):
        super().__init__()
        
        self.normal_prototypes_module = NormalPrototypes(
            num_prototypes=num_normal_prototypes,
            feature_dim=point_feat_dim
        )
        
        self.outlier_aggregator = OutlierFeatureAggregator(
            num_defect_tokens=num_defect_tokens,
            top_k=top_k_outliers,
            point_feat_dim=point_feat_dim,
            text_embed_dim=text_embed_dim,
            nhead=nhead
        )

    def forward(self, per_point_features: Tensor) -> Tensor:
        outlier_scores = self.normal_prototypes_module(per_point_features)

        defect_prompt_tokens = self.outlier_aggregator(per_point_features, outlier_scores)
        
        return defect_prompt_tokens



if __name__ == '__main__':
    p = torch.rand(4,3000,128)
    model = DefectPromptGenerator()
    out = model(p)
    print(out.shape)