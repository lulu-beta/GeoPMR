
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Dict
import math


class PoincareBall(nn.Module):
    """Poincaré Ball model for hyperbolic space operations."""
    
    def __init__(self, dim: int, c: float = 1.0, eps: float = 1e-5):
        super().__init__()
        self.dim = dim
        self.c = c
        self.eps = eps
    
    def distance(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute hyperbolic distance between two points."""
        x = self.project(x)
        y = self.project(y)
        
        diff_norm_sq = torch.sum((x - y) ** 2, dim=-1)
        x_norm_sq = torch.sum(x ** 2, dim=-1)
        y_norm_sq = torch.sum(y ** 2, dim=-1)
        
        denominator = (1 - x_norm_sq) * (1 - y_norm_sq)
        arcosh_arg = 1 + 2 * diff_norm_sq / (denominator + self.eps)
        arcosh_arg = torch.clamp(arcosh_arg, min=1.0 + self.eps)
        
        dist = torch.acosh(arcosh_arg) / math.sqrt(self.c)
        return dist
    
    def project(self, x: torch.Tensor) -> torch.Tensor:
        """Project points to Poincaré ball."""
        norm = torch.norm(x, dim=-1, keepdim=True)
        max_norm = 1 - self.eps
        cond = norm > max_norm
        x_proj = torch.where(cond, x * max_norm / (norm + self.eps), x)
        return x_proj
    
    def exp_map(self, v: torch.Tensor, x: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Exponential map: project from tangent space to hyperbolic space."""
        if x is None:
            x = torch.zeros_like(v)
            if v.dim() == 2:
                x = torch.zeros(v.shape[0], v.shape[1], device=v.device)
        
        v_norm = torch.norm(v, dim=-1, keepdim=True) + self.eps
        sqrt_c = math.sqrt(self.c)
        coeff = torch.tanh(sqrt_c * v_norm) / (sqrt_c * v_norm)
        v_scaled = coeff * v
        return self.mobius_add(x, v_scaled)
    
    def mobius_add(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Möbius addition in Poincaré ball."""
        x_norm_sq = torch.sum(x ** 2, dim=-1, keepdim=True)
        y_norm_sq = torch.sum(y ** 2, dim=-1, keepdim=True)
        xy = torch.sum(x * y, dim=-1, keepdim=True)
        
        numerator = (1 + 2 * xy + y_norm_sq) * x + (1 - x_norm_sq) * y
        denominator = 1 + 2 * xy + x_norm_sq * y_norm_sq
        
        result = numerator / (denominator + self.eps)
        return self.project(result)


class ProbabilisticHierarchyNode(nn.Module):
    """
    Probabilistic hierarchy node modeled as Gaussian distribution.
    - Leaf nodes: Single Gaussian N(μ, Σ)
    - Parent nodes: Mixture of Gaussians (MoG) of children
    """
    
    def __init__(self, dim: int, is_leaf: bool = True, num_children: int = 0):
        super().__init__()
        self.dim = dim
        self.is_leaf = is_leaf
        self.num_children = num_children
        
        if is_leaf:
            # Leaf node: learnable mean and covariance
            self.mean = nn.Parameter(torch.randn(1, dim) * 0.01)
            self.log_cov = nn.Parameter(torch.zeros(1, dim))
        else:
            # Parent node: mixing weights for MoG
            self.mixing_weights = nn.Parameter(torch.ones(num_children) / num_children)
    
    def forward(self, children_means: Optional[List[torch.Tensor]] = None,
                children_covs: Optional[List[torch.Tensor]] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns mean and covariance.
        For leaf: returns learned parameters.
        For parent: returns MoG parameters from children.
        """
        if self.is_leaf:
            mean = self.mean
            cov = torch.exp(self.log_cov) + 1e-6
            return mean, cov
        else:
            # Parent node: Mixture of Gaussians
            if children_means is None or children_covs is None:
                raise ValueError("Parent nodes require children parameters")
            
            # Ensure children are on same device and stack them
            children_means_stack = torch.stack(children_means, dim=0)  # (num_children, 1, dim)
            children_covs_stack = torch.stack(children_covs, dim=0)  # (num_children, 1, dim)
            
            # Get device from children (should be CUDA)
            device = children_means_stack.device
            
            # Normalize mixing weights and ensure on correct device
            mixing_weights = self.mixing_weights.to(device)
            weights = F.softmax(mixing_weights, dim=0)
            
            # MoG mean: weighted average
            mean = torch.sum(weights.view(-1, 1, 1) * children_means_stack, dim=0)
            
            # MoG covariance: weighted average + variance of means
            mean_cov = torch.sum(weights.view(-1, 1, 1) * children_covs_stack, dim=0)
            
            # Add variance of means (law of total variance)
            mean_diff = children_means_stack - mean.unsqueeze(0)
            mean_var = torch.sum(weights.view(-1, 1, 1) * (mean_diff ** 2), dim=0)
            cov = mean_cov + mean_var
            
            return mean, cov


class HHM(nn.Module):
    """    
    Hierarchy: Molecule 1D → Molecule 2D → Molecule 3D → Gene → Cell → Gene Expression
    """
    
    def __init__(
        self,
        num_modalities: int = 6,
        latent_dim: int = 300,
        hierarchy_structure: Optional[List[int]] = None,
        temperature: float = 0.1,
        lambda_sim: float = 1.0,
        lambda_dissim: float = 0.5,
        c: float = 1.0,
    ):
        """
        Args:
            num_modalities: Number of modalities (6)
            latent_dim: Dimension of latent space
            hierarchy_structure: Hierarchy levels [0, 1, 2, 3, 4, 5]
            temperature: Temperature for contrastive loss
            lambda_sim: Weight for parent-child similarity
            lambda_dissim: Weight for sibling dissimilarity
            c: Curvature of hyperbolic space
        """
        super().__init__()
        self.num_modalities = num_modalities
        self.latent_dim = latent_dim
        self.temperature = temperature
        self.lambda_sim = lambda_sim
        self.lambda_dissim = lambda_dissim
        
        if hierarchy_structure is None:
            hierarchy_structure = list(range(num_modalities))
        self.hierarchy_structure = hierarchy_structure
        self.max_level = max(hierarchy_structure)
        
        # Hyperbolic space operations
        self.poincare = PoincareBall(dim=latent_dim, c=c)
        
        # 1. Hierarchy Tree Pre-definition: Create probabilistic nodes
        # Build hierarchy tree: each level has nodes
        self.hierarchy_tree = self._build_hierarchy_tree()
        
        # 2. Projection from Euclidean to Hyperbolic
        self.euclidean_to_hyperbolic = nn.ModuleList([
            nn.Sequential(
                nn.Linear(latent_dim, latent_dim),
                nn.LayerNorm(latent_dim),
                nn.GELU(),
                nn.Linear(latent_dim, latent_dim),
                nn.LayerNorm(latent_dim),
            ) for _ in range(num_modalities)
        ])
        
        # Learnable scale for hyperbolic projection
        self.hyperbolic_scale = nn.Parameter(torch.ones(num_modalities) * 0.1)
        
        # 3. Hierarchy Decomposition Network
        # Decomposes features into hierarchical components
        self.hierarchy_decomposer = nn.ModuleList([
            nn.Sequential(
                nn.Linear(latent_dim, latent_dim),
                nn.LayerNorm(latent_dim),
                nn.GELU(),
            ) for _ in range(num_modalities)
        ])
        
        # 4. Hierarchy Encoder
        # Encodes hierarchical relations into global features
        self.hierarchy_encoder = nn.Sequential(
            nn.Linear(latent_dim * num_modalities, latent_dim * 2),
            nn.LayerNorm(latent_dim * 2),
            nn.GELU(),
            nn.Linear(latent_dim * 2, latent_dim),
        )
    
    def _build_hierarchy_tree(self) -> nn.ModuleDict:
        """
        Pre-define hierarchy tree structure.
        Returns an nn.ModuleDict so all ProbabilisticHierarchyNode parameters are
        properly registered, trained by the optimizer, moved to the correct device,
        and saved in the state dict.
        """
        tree = {}
        for level in range(self.max_level + 1):
            modalities_at_level = [i for i, l in enumerate(self.hierarchy_structure) if l == level]
            nodes_at_level = nn.ModuleList()
            for mod_idx in modalities_at_level:
                if level == 0:
                    nodes_at_level.append(
                        ProbabilisticHierarchyNode(self.latent_dim, is_leaf=True)
                    )
                else:
                    nodes_at_level.append(
                        ProbabilisticHierarchyNode(self.latent_dim, is_leaf=False, num_children=1)
                    )
            tree[str(level)] = nodes_at_level
        return nn.ModuleDict(tree)
    
    def project_to_hyperbolic(self, x_euclidean: torch.Tensor, modality_idx: int) -> torch.Tensor:
        """Project Euclidean embedding to hyperbolic space."""
        x_proj = self.euclidean_to_hyperbolic[modality_idx](x_euclidean)
        x_norm = F.normalize(x_proj, p=2, dim=-1)
        scale = torch.sigmoid(self.hyperbolic_scale[modality_idx]) * 0.9
        x_scaled = scale * x_norm
        return self.poincare.project(x_scaled)
    
    def hierarchy_decomposition(
        self,
        modal_latents: List[torch.Tensor]
    ) -> Dict[int, List[torch.Tensor]]:
        """
        Hierarchy Decomposition: Break down feature maps into hierarchical components.
        
        Args:
            modal_latents: List of modality embeddings (B, dim)
        
        Returns:
            Dictionary mapping level -> list of decomposed features at that level
        """
        decomposed = {}
        
        for level in range(self.max_level + 1):
            # Get modalities at this level
            modalities_at_level = [i for i, l in enumerate(self.hierarchy_structure) if l == level]
            
            level_features = []
            for mod_idx in modalities_at_level:
                # Decompose feature into hierarchical component
                decomposed_feat = self.hierarchy_decomposer[mod_idx](modal_latents[mod_idx])
                level_features.append(decomposed_feat)
            
            decomposed[level] = level_features
        
        return decomposed
    
    def get_hierarchy_nodes_hyperbolic(
        self,
        batch_size: int,
        device: torch.device,
        modal_hyperbolic_embeds: List[torch.Tensor]
    ) -> Dict[int, List[torch.Tensor]]:
        """
        Get hierarchy node embeddings in hyperbolic space.
        Uses probabilistic modeling: leaf nodes are Gaussians, parents are MoG.
        """
        hierarchy_nodes = {}
        
        # Process from bottom to top (level 0 to max_level)
        for level in range(self.max_level + 1):
            nodes_at_level = []
            modalities_at_level = [i for i, l in enumerate(self.hierarchy_structure) if l == level]
            
            for idx, mod_idx in enumerate(modalities_at_level):
                node = self.hierarchy_tree[str(level)][idx]
                
                if level == 0:
                    # Leaf node (Molecule 1D): Gaussian distribution
                    mean, cov = node()
                    # Ensure mean and cov are on correct device
                    mean = mean.to(device)
                    cov = cov.to(device)
                    # Use mean with small noise for stability
                    noise = torch.randn(batch_size, self.latent_dim, device=device) * 0.01
                    node_embed = mean.expand(batch_size, -1) + noise
                    
                    # Project to hyperbolic space
                    node_hyperbolic = self.poincare.exp_map(node_embed * 0.1)
                    nodes_at_level.append(node_hyperbolic)
                else:
                    # Parent node: MoG of immediate child (previous level)
                    # Find child at level-1
                    child_level = level - 1
                    child_modalities = [i for i, l in enumerate(self.hierarchy_structure) if l == child_level]
                    
                    if child_modalities:
                        # Get child node (from previous level or from modal embeddings)
                        child_mod_idx = child_modalities[0]  # Use first child
                        
                        # Use the actual modal embedding as child mean
                        child_embed = modal_hyperbolic_embeds[child_mod_idx]  # (B, dim)
                        child_mean = child_embed.mean(dim=0, keepdim=True)  # (1, dim)
                        child_cov = torch.ones(1, self.latent_dim, device=device) * 0.01
                        
                        # Compute MoG
                        mean, cov = node([child_mean], [child_cov])
                        # Ensure mean is on correct device
                        mean = mean.to(device)
                        cov = cov.to(device)
                        
                        # Project to hyperbolic
                        node_embed = mean.expand(batch_size, -1)
                        node_hyperbolic = self.poincare.exp_map(node_embed * 0.1)
                        nodes_at_level.append(node_hyperbolic)
                    else:
                        # Fallback: use modal embedding directly
                        node_hyperbolic = modal_hyperbolic_embeds[mod_idx]
                        nodes_at_level.append(node_hyperbolic)
            
            hierarchy_nodes[level] = nodes_at_level
        
        return hierarchy_nodes
    
    def hierarchical_contrastive_loss(
        self,
        modal_hyperbolic_embeds: List[torch.Tensor],
        hierarchy_nodes: Dict[int, List[torch.Tensor]]
    ) -> torch.Tensor:
        """
        Hierarchical Contrastive Loss:
        - Child-parent nodes: similar (pull together)
        - Child-child nodes: dissimilar (push apart)
        """
        batch_size = modal_hyperbolic_embeds[0].shape[0]
        device = modal_hyperbolic_embeds[0].device
        
        # Parent-child similarity loss
        similarity_losses = []
        for i, level in enumerate(self.hierarchy_structure):
            child_embed = modal_hyperbolic_embeds[i]  # (B, dim)
            
            # Get parent node at this level
            if level in hierarchy_nodes and len(hierarchy_nodes[level]) > 0:
                # Find corresponding parent node
                parent_idx = 0  # Simplified: use first node at level
                if parent_idx < len(hierarchy_nodes[level]):
                    parent_node = hierarchy_nodes[level][parent_idx]  # (B, dim)
                    
                    # Compute hyperbolic distance
                    distances = self.poincare.distance(child_embed, parent_node)  # (B,)
                    
                    # Similarity loss: minimize distance
                    sim_loss = (distances ** 2).mean() / (self.temperature ** 2)
                    similarity_losses.append(sim_loss)
        
        similarity_loss = torch.stack(similarity_losses).mean() if similarity_losses else torch.tensor(0.0, device=device)
        
        # Sibling dissimilarity loss (adjacent modalities)
        dissimilarity_losses = []
        for i in range(len(modal_hyperbolic_embeds)):
            for j in range(i + 1, len(modal_hyperbolic_embeds)):
                level_i = self.hierarchy_structure[i]
                level_j = self.hierarchy_structure[j]
                
                # Adjacent modalities (hierarchical neighbors)
                if abs(level_i - level_j) == 1:
                    sibling_i = modal_hyperbolic_embeds[i]  # (B, dim)
                    sibling_j = modal_hyperbolic_embeds[j]  # (B, dim)
                    
                    distances = self.poincare.distance(sibling_i, sibling_j)  # (B,)
                    
                    # Dissimilarity: encourage minimum distance
                    margin = 0.3
                    dissim_loss = F.softplus(margin - distances).mean()
                    dissimilarity_losses.append(dissim_loss)
        
        if len(dissimilarity_losses) > 0:
            dissimilarity_loss = torch.stack(dissimilarity_losses).mean()
        else:
            dissimilarity_loss = torch.tensor(0.0, device=device)
        
        # Combine losses
        total_loss = (
            self.lambda_sim * similarity_loss +
            self.lambda_dissim * dissimilarity_loss
        )
        
        return total_loss
    
    def hierarchy_encoding(
        self,
        modal_hyperbolic_embeds: List[torch.Tensor]
    ) -> torch.Tensor:
        """
        Hierarchy Encoding: Encode hierarchical relations into global features.
        """
        # Concatenate all modality embeddings
        all_embeds = torch.cat(modal_hyperbolic_embeds, dim=-1)  # (B, num_modalities * dim)
        
        # Encode through MLP
        encoded = self.hierarchy_encoder(all_embeds)  # (B, dim)
        
        return encoded
    
    def forward(
        self,
        modal_latents: List[torch.Tensor]
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], torch.Tensor]:
        """
        Complete workflow:
        1. Pre-define hierarchy tree (done in __init__)
        2. Hyperbolic mapping
        3. Hierarchical contrastive learning
        4. Hierarchy encoding
        """
        batch_size = modal_latents[0].shape[0]
        device = modal_latents[0].device
        
        # 2. Hyperbolic Mapping: Project to hyperbolic space
        modal_hyperbolic_embeds = []
        for i, latent in enumerate(modal_latents):
            h_embed = self.project_to_hyperbolic(latent, i)
            modal_hyperbolic_embeds.append(h_embed)
        
        # Get hierarchy nodes in hyperbolic space (with probabilistic modeling)
        hierarchy_nodes = self.get_hierarchy_nodes_hyperbolic(
            batch_size, device, modal_hyperbolic_embeds
        )
        
        # 3. Hierarchical Contrastive Learning
        total_loss = self.hierarchical_contrastive_loss(
            modal_hyperbolic_embeds,
            hierarchy_nodes
        )
        
        # 4. Hierarchy Encoding (for future use)
        encoded_features = self.hierarchy_encoding(modal_hyperbolic_embeds)
        
        # Create output dictionaries (for compatibility)
        latent_codes = {}
        latent_vectors = {}
        
        for i in range(self.num_modalities):
            level = self.hierarchy_structure[i]
            latent_codes[f'm{i}'] = torch.full(
                (batch_size, 1), level, dtype=torch.long, device=device
            )
            latent_vectors[f'm{i}'] = modal_hyperbolic_embeds[i].unsqueeze(1)  # (B, 1, dim)
        
        return latent_codes, latent_vectors, total_loss
