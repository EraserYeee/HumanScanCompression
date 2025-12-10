import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_max

class ScatterSTNkd(nn.Module):
    """
    Scatter version of STNkd (Spatial Transformer Network for k-dim features).
    Adapts PointNet's T-Net for flattened, variable-size point clusters.
    """
    def __init__(self, k=64):
        super().__init__()
        self.k = k
        
        # Shared MLPs (applied to each point)
        # Input: (k) -> 64 -> 128 -> 1024
        self.conv1 = nn.Linear(k, 64)
        self.conv2 = nn.Linear(64, 128)
        self.conv3 = nn.Linear(128, 1024)
        
        # Global MLPs (applied to global feature per cluster)
        # Input: (1024) -> 512 -> 256 -> k*k
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)
        
        # BatchNorms
        # Note: BatchNorm1d supports (N, C) input directly
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x, cluster_idx, total_clusters):
        """
        Args:
            x: (Total_Points, k) Feature vectors for all points in batch
            cluster_idx: (Total_Points,) Global cluster index for each point
            total_clusters: int (B * V), Total number of clusters (vertices)
            
        Returns:
            x_transformed: (Total_Points, k)
            trans: (B*V, k, k) Transformation matrices
        """
        # 1. Point-wise feature extraction
        x_local = F.relu(self.bn1(self.conv1(x)))
        x_local = F.relu(self.bn2(self.conv2(x_local)))
        x_local = F.relu(self.bn3(self.conv3(x_local))) # (Total_Points, 1024)
        
        # 2. Scatter Max Pool -> Global Feature (per cluster)
        # global_feat: (B*V, 1024)
        # Fill with 0 if cluster is empty (though unlikely in training if sampled correctly)
        global_feat, _ = scatter_max(x_local, cluster_idx, dim=0, dim_size=total_clusters)
        
        # 3. Global MLP -> Matrix
        g = F.relu(self.bn4(self.fc1(global_feat)))
        g = F.relu(self.bn5(self.fc2(g)))
        trans = self.fc3(g) # (B*V, k*k)
        
        # 4. Reshape & Identity Add
        trans = trans.view(-1, self.k, self.k)
        
        # Initialize as Identity + small noise (handled by weights) 
        # But standard practice is adding Identity explicitly
        iden = torch.eye(self.k, device=x.device).view(1, self.k, self.k)
        trans = trans + iden # (B*V, k, k)
        
        # 5. Broadcast back to points & Matmul
        # We need to apply T_i to all points belonging to cluster i
        
        # Expand trans for each point: (Total_Points, k, k)
        # This uses the cluster_idx to look up the matrix for each point
        trans_expanded = trans[cluster_idx] 
        
        # Matrix Multiplication
        # x: (Total_Points, k) -> (Total_Points, 1, k)
        # trans: (Total_Points, k, k)
        # result: (1, k) @ (k, k) -> (1, k)
        x_transformed = torch.matmul(x.unsqueeze(1), trans_expanded).squeeze(1)
        
        return x_transformed, trans

def feature_transform_regularizer(trans):
    """
    Regularization loss to encourage orthogonality.
    Force trans * trans^T to be Identity.
    trans: (B, K, K)
    """
    d = trans.size()[1]
    I = torch.eye(d, device=trans.device)[None, :, :]
    
    # || T * T^T - I ||^2
    loss = torch.mean(torch.norm(torch.bmm(trans, trans.transpose(2, 1)) - I, dim=(1, 2)))
    return loss

