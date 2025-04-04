import math
import torch
from torch import nn

CONNECTIONS = {
    0: [1, 2, 3],        # Pelvis to Left Hip, Right Hip, Spine 1
    1: [0, 4],           # Left Hip to Pelvis, Left Knee
    2: [0, 5],           # Right Hip to Pelvis, Right Knee
    3: [0, 6],           # Spine 1 to Pelvis, Spine 2
    4: [1, 7],           # Left Knee to Left Hip, Left Ankle
    5: [2, 8],           # Right Knee to Right Hip, Right Ankle
    6: [3, 9],           # Spine 2 to Spine 1, Spine 3
    7: [4, 10],          # Left Ankle to Left Knee, Left Foot
    8: [5, 11],          # Right Ankle to Right Knee, Right Foot
    9: [6, 12, 13, 14],  # Spine 3 to Spine 2, Neck, Left Collar, Right Collar
    10: [7],             # Left Foot to Left Ankle
    11: [8],             # Right Foot to Right Ankle
    12: [9, 15],         # Neck to Spine 3, Head
    13: [9, 16],         # Left Collar to Spine 3, Left Shoulder
    14: [9, 17],         # Right Collar to Spine 3, Right Shoulder
    15: [12],            # Head to Neck
    16: [13, 18],        # Left Shoulder to Left Collar, Left Elbow
    17: [14, 19],        # Right Shoulder to Right Collar, Right Elbow
    18: [16, 20],        # Left Elbow to Left Shoulder, Left Wrist
    19: [17, 21],        # Right Elbow to Right Shoulder, Right Wrist
    20: [18, 22],        # Left Wrist to Left Elbow, Left Hand
    21: [19, 23],        # Right Wrist to Right Elbow, Right Hand
    22: [20],            # Left Hand to Left Wrist
    23: [21]             # Right Hand to Right Wrist
}
# Connections based on SMPL with 24 joints

class GCN(nn.Module):
    def __init__(self, dim_in, dim_out, num_nodes, neighbour_num=4, mode='spatial',
                 use_temporal_similarity=True, temporal_connection_len=1, connections=None):
        """
        :param dim_in: Input channel dimension.
        :param dim_out: Output channel dimension.
        :param num_nodes: Number of nodes (for spatial, e.g., 24).
        :param neighbour_num: Used in temporal GCN to select top-k similar frames.
        :param mode: 'spatial' or 'temporal'.
        :param use_temporal_similarity: If True, compute a dynamic similarity-based adjacency.
        :param temporal_connection_len: Temporal connection window for fixed adjacency (if not using similarity).
        :param connections: For spatial mode, the predefined graph connections.
        """
        super().__init__()
        assert mode in ['spatial', 'temporal'], "Mode must be 'spatial' or 'temporal'."

        self.relu = nn.ReLU()
        self.neighbour_num = neighbour_num
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.mode = mode
        self.use_temporal_similarity = use_temporal_similarity
        self.num_nodes = num_nodes
        self.connections = connections
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.U = nn.Linear(self.dim_in, self.dim_out)
        self.V = nn.Linear(self.dim_in, self.dim_out)
        # Change BN to normalize over feature channels (dim_out) instead of num_nodes.
        self.batch_norm = nn.BatchNorm1d(self.dim_out)

        self._init_gcn()

        if mode == 'spatial':
            self.adj = self._init_spatial_adj()
        elif mode == 'temporal' and not self.use_temporal_similarity:
            self.fixed_temporal_connection_len = temporal_connection_len

    def _init_gcn(self):
        self.U.weight.data.normal_(0, math.sqrt(2. / self.dim_in))
        self.V.weight.data.normal_(0, math.sqrt(2. / self.dim_in))
        # BatchNorm parameters are now for self.dim_out channels.
        self.batch_norm.weight.data.fill_(1)
        self.batch_norm.bias.data.zero_()

    def _init_spatial_adj(self):
        adj = torch.zeros((self.num_nodes, self.num_nodes))
        connections = self.connections if self.connections is not None else CONNECTIONS
        for i in range(self.num_nodes):
            connected_nodes = connections.get(i, [])
            for j in connected_nodes:
                adj[i, j] = 1
                adj[j, i] = 1  # symmetric
        return adj

    def _init_temporal_adj_dynamic(self, T):
        """
        Dynamically create a fixed temporal adjacency matrix for sequence length T.
        Each time step is connected to itself and the next `fixed_temporal_connection_len` steps.
        """
        adj = torch.zeros((T, T))
        for i in range(T):
            for offset in range(self.fixed_temporal_connection_len + 1):
                if i + offset < T:
                    adj[i, i + offset] = 1
                    adj[i + offset, i] = 1  # symmetric
        return adj

    @staticmethod
    def normalize_digraph(adj):
        b, n, c = adj.shape
        node_degrees = adj.detach().sum(dim=-1)
        deg_inv_sqrt = node_degrees.pow(-0.5)
        norm_deg_matrix = torch.stack([torch.diag(d) for d in deg_inv_sqrt])
        norm_deg_matrix = norm_deg_matrix.to(adj.device)
        norm_adj = torch.bmm(torch.bmm(norm_deg_matrix, adj), norm_deg_matrix)
        return norm_adj

    def change_adj_device_to_cuda(self, adj):
        dev = self.V.weight.get_device()
        if dev >= 0 and adj.get_device() < 0:
            adj = adj.to(dev)
        return adj

    def forward(self, x, mask=None):
        """
        Args:
            x: Tensor with shape [B, T, J, C].
            mask: Optional tensor with shape [B, T] (1 for valid frames, 0 for padded),
                  only used in temporal mode when using similarity-based adjacency.
        Returns:
            Tensor of shape [B, T, J, dim_out] (for temporal mode, reshaped appropriately).
        """
        b, t, j, c = x.shape
        if self.mode == 'temporal':
            # Rearrange x: [B, T, J, C] -> [B, J, T, C] then flatten to [B*J, T, C]
            x = x.transpose(1, 2)  # [B, J, T, C]
            x = x.reshape(b * j, t, c)  # [B*J, T, C]
            if self.use_temporal_similarity:
                similarity = x @ x.transpose(1, 2)  # [B*J, T, T]
                if mask is not None:
                    # Expand mask from [B, T] to [B, 1, T] and then to [B*J, T]
                    mask_expanded = mask.unsqueeze(1).repeat(1, j, 1).reshape(b * j, t).to(self.device)
                    similarity = similarity.masked_fill(mask_expanded.unsqueeze(1) == 0, float('-inf'))
                threshold = similarity.topk(k=self.neighbour_num, dim=-1, largest=True)[0][..., -1].view(b * j, t, 1)
                adj = (similarity >= threshold).float()
            else:
                # Use fixed dynamic adjacency based on input T
                adj = self._init_temporal_adj_dynamic(t)  # [T, T]
                adj = self.change_adj_device_to_cuda(adj)
                adj = adj.repeat(b * j, 1, 1)
        else:
            # Spatial mode: reshape x to [B*T, J, C]
            x = x.reshape(b * t, j, c)
            adj = self.adj  # [num_nodes, num_nodes]
            adj = self.change_adj_device_to_cuda(adj)
            adj = adj.repeat(b * t, 1, 1)

        norm_adj = self.normalize_digraph(adj)
        aggregate = norm_adj @ self.V(x)  # [*, ?, dim_out]

        # Compute (aggregate + self.U(x)) then apply BatchNorm on the feature dimension.
        out = aggregate + self.U(x)  # shape: [*, J, dim_out] or [*, T, dim_out] in temporal mode.
        # Transpose to [*, dim_out, J] so that BN (with num_features=dim_out) is applied along the channel dimension.
        out = out.transpose(1, 2)  # [*, dim_out, J]
        out = self.batch_norm(out)  # BN over dim_out channels
        out = out.transpose(1, 2)  # back to [*, J, dim_out]

        if self.dim_in == self.dim_out:
            x = self.relu(x + out)
        else:
            x = self.relu(out)
        
        if self.mode == 'spatial':
            x = x.reshape(b, t, j, self.dim_out)
        else:
            # For temporal, x is [B*J, T, dim_out] -> reshape to [B, J, T, dim_out] and then transpose to [B, T, J, dim_out]
            x = x.reshape(b, j, t, self.dim_out).transpose(1, 2)
        return x
