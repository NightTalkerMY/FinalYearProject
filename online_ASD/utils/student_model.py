import torch
import torch.nn as nn
import torch.nn.functional as F

class CausalGraphNet(nn.Module):
    """
    A pure PyTorch implementation of the Causal MSSG Graph.
    Avoids torch_geometric dependencies for easier Raspberry Pi 5 deployment.
    """
    def __init__(self, in_features=128, out_features=128):
        super(CausalGraphNet, self).__init__()
        # The Graph Convolution weight matrix
        self.weight = nn.Parameter(torch.Tensor(in_features, out_features))
        nn.init.xavier_uniform_(self.weight)
        
    def forward(self, x, mask):
        # x shape: [Batch, Speakers(5), Frames(50), 128]
        # mask shape: [Batch, Speakers(5), Frames(50)]
        B, S, T, D = x.shape
        
        # Flatten spatial and temporal dimensions to create nodes: [B, S*T, 128]
        x_flat = x.view(B, S * T, D)
        mask_flat = mask.view(B, S * T)
        
        # 1. Initialize an empty Adjacency Matrix A [B, S*T, S*T]
        # A will dictate which nodes can "talk" to which other nodes.
        A = torch.zeros((B, S * T, S * T), device=x.device)
        
        # 2. Build the Causal Edges (Intra-speaker and Inter-speaker)
        for s in range(S):
            for t in range(T):
                current_node = s * T + t
                
                # A. Intra-speaker (Temporal): Connect to own past frames
                # Strict Causal Rule: t_prev must be <= t
                for t_prev in range(max(0, t - 3), t + 1): # Look back up to 3 frames
                    past_node = s * T + t_prev
                    A[:, current_node, past_node] = 1.0
                    
                # B. Inter-speaker (Spatial): Connect to other speakers in the EXACT SAME frame
                for other_s in range(S):
                    if other_s != s:
                        other_node = other_s * T + t
                        A[:, current_node, other_node] = 1.0

        # 3. Apply the Ghost Hunter Mask!
        # If a node is a ghost (mask == 0), sever all connections to it
        valid_matrix = mask_flat.unsqueeze(1) * mask_flat.unsqueeze(2) # [B, S*T, S*T]
        A = A * valid_matrix
        
        # Normalize the adjacency matrix to prevent exploding gradients
        row_sum = A.sum(dim=-1, keepdim=True).clamp(min=1e-9)
        A_norm = A / row_sum
        
        # 4. Perform the Graph Convolution: Output = A_norm * (X * W)
        node_transform = torch.matmul(x_flat, self.weight) # [B, S*T, 128]
        graph_out = torch.bmm(A_norm, node_transform)      # [B, S*T, 128]
        
        # Reshape back to our 4D structure and add a residual connection
        graph_out = graph_out.view(B, S, T, D)
        return F.relu(graph_out + x)

class CausalStudentASD(nn.Module):
    def __init__(self, visual_dim=896, hidden_dim=128, num_classes=2):
        super(CausalStudentASD, self).__init__()
        
        # 1. Visual Compression Layer
        # Shrinks the 896 concatenated visual features down to 128 to match the audio
        self.v_proj = nn.Linear(visual_dim, hidden_dim)
        
        # 2. The Causal Sequence Module
        # batch_first=True, bidirectional=False (Crucial for real-time causality)
        self.gru = nn.GRU(hidden_dim, hidden_dim, num_layers=1, batch_first=True, bidirectional=False)
        
        # Buffer Attention (Multihead)
        self.mha = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=4, batch_first=True)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
        # 3. The Causal Graph
        self.causal_graph = CausalGraphNet(in_features=hidden_dim, out_features=hidden_dim)
        
        # 4. The Final Classifier
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, v, a, s_cues, mask):
        """
        v: Visual features [B, 5, 50, 896]
        a: Audio features [B, 5, 50, 128]
        s_cues: Spatial info [B, 5, 50, 4] (Passed for future architectural expansion)
        mask: Valid node mask [B, 5, 50]
        """
        B, S, T, _ = v.shape
        
        # Step 1: Compress Visuals
        v_compressed = F.relu(self.v_proj(v)) # [B, S, T, 128]
        
        # Step 2: Modality Fusion (Simple Addition, fast for Raspberry Pi)
        x = v_compressed + a # [B, S, T, 128]
        
        # Step 3: Sequence Processing (Process each speaker's timeline independently)
        # Reshape to [B * S, T, 128] so the GRU processes 5 timelines in parallel
        x_seq = x.view(B * S, T, 128)

        gru_out, _ = self.gru(x_seq) # gru_out: [B * S, T, 128]

        # Step 4: Buffer Attention with Strict Causal Mask
        # The mask forces the attention layer to ignore future frames
        causal_mask = torch.triu(torch.ones(T, T) * float('-inf'), diagonal=1).to(v.device)

        attn_out, _ = self.mha(gru_out, gru_out, gru_out, attn_mask=causal_mask)
        seq_features = self.layer_norm(gru_out + attn_out) # Residual connection

        # Reshape back to the scene matrix
        seq_features = seq_features.view(B, S, T, 128)

        # Zero out ghost speaker outputs so GRU artifacts don't leak into the graph
        seq_features = seq_features * mask.unsqueeze(-1)  # mask: [B,S,T] → [B,S,T,1]
        
        # Step 5: Inter/Intra Speaker Graph Routing
        # This is where the student mimics the Teacher's MSSG behavior
        graph_features = self.causal_graph(seq_features, mask)

        # Step 6: Classification
        logits = self.classifier(graph_features) # [B, S, T, 2]

        # Return logits (for CE) and pre-graph features (for KD MSE against teacher_av,
        # which is also a pre-graph representation from the teacher's BiGRU output)
        return logits, seq_features