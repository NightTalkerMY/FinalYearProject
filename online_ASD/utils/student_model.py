import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossSpeakerAttention(nn.Module):
    """
    Replaces CausalGraphNet. At each timestep, speakers attend to each other
    via multi-head attention — learns WHO to attend to and WHAT to extract.
    Ghost speakers are masked out.
    """
    def __init__(self, hidden_dim=128, num_heads=4, dropout=0.1):
        super().__init__()
        self.mha = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads,
            batch_first=True, dropout=dropout
        )
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x, mask):
        # x: [B, S, T, D], mask: [B, S, T]
        B, S, T, D = x.shape

        # Reshape to [B*T, S, D] — at each timestep, speakers attend to each other
        x_cross = x.permute(0, 2, 1, 3).reshape(B * T, S, D)

        # Build key_padding_mask: [B*T, S] — True means IGNORE this speaker
        mask_cross = mask.permute(0, 2, 1).reshape(B * T, S)  # [B*T, S]
        key_padding_mask = mask_cross < 0.5  # True = ghost = ignore

        # If all speakers are ghosts at a timestep, skip masking to avoid NaN
        all_masked = key_padding_mask.all(dim=-1)  # [B*T]
        key_padding_mask[all_masked] = False  # allow attention (features are zero anyway)

        attn_out, _ = self.mha(
            x_cross, x_cross, x_cross,
            key_padding_mask=key_padding_mask
        )
        out = self.layer_norm(x_cross + attn_out)

        # Reshape back to [B, S, T, D]
        out = out.reshape(B, T, S, D).permute(0, 2, 1, 3)
        return out


class CausalStudentASD(nn.Module):
    NUM_SCALES = 7
    SCALE_DIM = 128  # Each of the 7 visual scales is 128-dim in the cache

    def __init__(self, hidden_dim=128, num_classes=2):
        super(CausalStudentASD, self).__init__()

        # 1. Per-scale visual projections (preserves scale independence)
        self.scale_projs = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.SCALE_DIM, 64),
                nn.ReLU(),
                nn.Dropout(0.1),
            )
            for _ in range(self.NUM_SCALES)
        ])

        # 2. Spatial-aware scale attention
        self.scale_attn = nn.Sequential(
            nn.Linear(4, 32),
            nn.ReLU(),
            nn.Linear(32, self.NUM_SCALES),
        )

        # 3. Scale fusion → hidden_dim
        self.fuse = nn.Sequential(
            nn.Linear(64, hidden_dim),
            nn.ReLU(),
        )

        # 4. Causal GRU (2-layer with dropout)
        self.gru = nn.GRU(hidden_dim, hidden_dim, num_layers=2,
                          batch_first=True, bidirectional=False, dropout=0.1)

        # 5. Buffer Attention (Multihead with dropout)
        self.mha = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=4,
                                         batch_first=True, dropout=0.1)
        self.temporal_norm = nn.LayerNorm(hidden_dim)

        # 6. Gated FFN — mimics teacher's backward GRU gated feedforward transform
        self.gate_proj = nn.Linear(hidden_dim, hidden_dim)
        self.value_proj = nn.Linear(hidden_dim, hidden_dim)
        self.ffn_norm = nn.LayerNorm(hidden_dim)

        # 7. Cross-Speaker Attention (replaces CausalGraphNet)
        self.cross_speaker = CrossSpeakerAttention(hidden_dim, num_heads=4, dropout=0.1)

        # 8. The Final Classifier (with dropout)
        self.feat_dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, v, a, s_cues, mask):
        """
        v: Visual features [B, 5, 50, 896] — 7 scales × 128-dim concatenated
        a: Audio features [B, 5, 50, 128]
        s_cues: Spatial info [B, 5, 50, 4] — face position/size [cx, cy, w, h]
        mask: Valid node mask [B, 5, 50]
        """
        B, S, T, _ = v.shape

        # Step 1: Per-scale visual processing
        scale_feats = []
        for i, proj in enumerate(self.scale_projs):
            scale_v = v[:, :, :, i * self.SCALE_DIM:(i + 1) * self.SCALE_DIM]
            scale_feats.append(proj(scale_v))
        scale_stack = torch.stack(scale_feats, dim=3)  # [B,S,T,7,64]

        # Step 2: Spatial-aware scale attention
        attn_weights = F.softmax(self.scale_attn(s_cues), dim=-1)  # [B,S,T,7]
        v_fused = (scale_stack * attn_weights.unsqueeze(-1)).sum(dim=3)  # [B,S,T,64]

        # Step 3: Fuse to hidden_dim + audio
        x = self.fuse(v_fused) + a  # [B,S,T,128]

        # Step 4: Causal temporal processing (each speaker independently)
        D = x.shape[-1]
        x_seq = x.view(B * S, T, D)
        gru_out, _ = self.gru(x_seq)  # [B*S, T, 128]

        # Step 5: Buffer Attention with Strict Causal Mask
        causal_mask = torch.triu(torch.ones(T, T, device=v.device) * float('-inf'), diagonal=1)
        attn_out, _ = self.mha(gru_out, gru_out, gru_out, attn_mask=causal_mask)
        seq_features = self.temporal_norm(gru_out + attn_out)

        # Step 6: Gated FFN — backward GRU equivalent
        gate = torch.sigmoid(self.gate_proj(seq_features))
        value = torch.tanh(self.value_proj(seq_features))
        seq_features = self.ffn_norm(seq_features + gate * value)

        # Reshape back to the scene matrix
        seq_features = seq_features.view(B, S, T, D)

        # Zero out ghost speaker outputs
        seq_features = seq_features * mask.unsqueeze(-1)

        # Step 7: Cross-Speaker Attention (replaces graph)
        graph_features = self.cross_speaker(seq_features, mask)

        # Step 8: Classification
        logits = self.classifier(self.feat_dropout(graph_features))  # [B, S, T, 2]

        # Return logits (for CE) and pre-graph features (for KD MSE)
        return logits, seq_features