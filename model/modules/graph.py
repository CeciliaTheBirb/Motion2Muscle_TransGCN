import math
import torch
from torch import nn

CONNECTIONS = {
    0: [1, 2, 3],         # Pelvis → Left Hip, Right Hip, Spine 1
    1: [0, 4],            # Left Hip → Pelvis, Left Knee
    2: [0, 5],            # Right Hip → Pelvis, Right Knee
    3: [0, 6],            # Spine 1 → Pelvis, Spine 2
    4: [1, 7],            # Left Knee → Left Hip, Left Ankle
    5: [2, 8],            # Right Knee → Right Hip, Right Ankle
    6: [3, 9],            # Spine 2 → Spine 1, Spine 3
    7: [4],               # Left Ankle → Left Knee
    8: [5],               # Right Ankle → Right Knee
    9: [6, 10, 11, 12],   # Spine 3 → Spine 2, Neck, Left Collar, Right Collar
    10: [9, 13],          # Neck → Spine 3, Head
    11: [9, 14],          # Left Collar → Spine 3, Left Shoulder
    12: [9, 15],          # Right Collar → Spine 3, Right Shoulder
    13: [10],             # Head → Neck
    14: [11, 16],         # Left Shoulder → Left Collar, Left Elbow
    15: [12, 17],         # Right Shoulder → Right Collar, Right Elbow
    16: [14, 18],         # Left Elbow → Left Shoulder, Left Wrist
    17: [15, 19],         # Right Elbow → Right Shoulder, Right Wrist
    18: [16],             # Left Wrist → Left Elbow
    19: [17],             # Right Wrist → Right Elbow
}

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
        self.num_nodes = 20
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
                adj[j, i] = 1  
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
                    adj[i + offset, i] = 1  
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
            x = x.transpose(1, 2) 
            x = x.reshape(b * j, t, c)
            if self.use_temporal_similarity:
                similarity = x @ x.transpose(1, 2) 
                threshold = similarity.topk(k=self.neighbour_num, dim=-1, largest=True)[0][..., -1].view(b * j, t, 1)
                adj = (similarity >= threshold).float()
            else:
                adj = self._init_temporal_adj_dynamic(t)  
                adj = self.change_adj_device_to_cuda(adj)
                adj = adj.repeat(b * j, 1, 1)
        else:
            x = x.reshape(b * t, j, c)
            adj = self.adj 
            adj = self.change_adj_device_to_cuda(adj)
            adj = adj.repeat(b * t, 1, 1)

        norm_adj = self.normalize_digraph(adj)
        aggregate = norm_adj @ self.V(x)  
        out = aggregate + self.U(x) 
        out = out.transpose(1, 2) 
        out = self.batch_norm(out) 
        out = out.transpose(1, 2) 

        if self.dim_in == self.dim_out:
            x = self.relu(x + out)
        else:
            x = self.relu(out)
        
        if self.mode == 'spatial':
            x = x.reshape(b, t, j, self.dim_out)
        else:
            x = x.reshape(b, j, t, self.dim_out).transpose(1, 2)
        return x
