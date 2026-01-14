
import torch
import torch.nn as nn
import torch.nn.functional as F

def kmeans(data, num_clusters, num_iters=10):
    """Simple k-means clustering implementation."""
    batch_size = data.size(0)
    device = data.device

    # Random initialization
    indices = torch.randperm(batch_size, device=device)[:num_clusters]
    centers = data[indices].clone()

    for _ in range(num_iters):
        # Compute L2 distances
        d = torch.sum(data ** 2, dim=1, keepdim=True) + \
            torch.sum(centers ** 2, dim=1, keepdim=True).t() - \
            2 * torch.matmul(data, centers.t())

        # Assign to nearest center
        assignments = torch.argmin(d, dim=1)

        # Update centers
        new_centers = torch.zeros_like(centers)
        counts = torch.zeros(num_clusters, device=device)
        for i in range(num_clusters):
            mask = (assignments == i)
            if mask.sum() > 0:
                new_centers[i] = data[mask].mean(dim=0)
                counts[i] = mask.sum()
            else:
                # Keep old center if no points assigned
                new_centers[i] = centers[i]

        centers = new_centers

    return centers


class VectorQuantizerEMA(nn.Module):
    """EMA-based Vector Quantizer with Cosine Similarity.

    Ablation switches:
        - use_gradient_update: Use k-means initialization + gradient update instead of EMA.
        - use_euclidean: Use Euclidean distance instead of cosine similarity.
    """

    def __init__(self, n_e, e_dim, beta=0.25, decay=0.99, epsilon=1e-5,
                 restart_threshold=1.0, use_codebook_restart=True,
                 use_gradient_update=False, use_euclidean=False, kmeans_iters=100):
        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta
        self.decay = decay
        self.epsilon = epsilon
        self.restart_threshold = restart_threshold
        self.use_codebook_restart = use_codebook_restart
        self.use_gradient_update = use_gradient_update
        self.use_euclidean = use_euclidean
        self.kmeans_iters = kmeans_iters

        self.embedding = nn.Embedding(self.n_e, self.e_dim)

        if use_gradient_update:
            # K-means init mode: zero initialize, will be set during first forward
            self.embedding.weight.data.zero_()
            self.register_buffer('_initted', torch.tensor(False))
        else:
            # EMA mode: normal random init
            self.embedding.weight.data.normal_()
            self.register_buffer('_initted', torch.tensor(True))

        # EMA buffers (only used when use_gradient_update=False)
        self.register_buffer('ema_cluster_size', torch.zeros(n_e))
        if not use_gradient_update:
            self.register_buffer('ema_w', torch.Tensor(n_e, e_dim))
            self.ema_w.data.normal_()
        else:
            self.register_buffer('ema_w', torch.zeros(n_e, e_dim))

    def _init_codebook(self, z):
        """Initialize codebook with k-means clustering."""
        if self._initted:
            return

        centers = kmeans(z, self.n_e, self.kmeans_iters)
        self.embedding.weight.data.copy_(centers)
        self._initted.fill_(True)

    def _restart_dead_codes(self, z, cluster_size):
        """Restart dead codebook entries with diversity-aware sampling."""
        dead_codes = cluster_size < self.restart_threshold
        if not dead_codes.any():
            return

        n_dead = dead_codes.sum().item()
        z_size = z.size(0)

        if n_dead > z_size:
            rand_idx = torch.cat([
                torch.randperm(z_size, device=z.device),
                torch.randint(0, z_size, (n_dead - z_size,), device=z.device)
            ])
        else:
            with torch.no_grad():
                active_codes = self.embedding.weight.data[~dead_codes]
                if active_codes.size(0) > 0:
                    if self.use_euclidean:
                        # Euclidean distance for diversity sampling
                        d = torch.sum(z ** 2, dim=1, keepdim=True) + \
                            torch.sum(active_codes ** 2, dim=1, keepdim=True).t() - \
                            2 * torch.matmul(z, active_codes.t())
                        min_dist, _ = d.min(dim=1)
                    else:
                        # Cosine distance for diversity sampling
                        z_norm = F.normalize(z, dim=1)
                        a_norm = F.normalize(active_codes, dim=1)
                        d = 1 - torch.matmul(z_norm, a_norm.t())
                        min_dist, _ = d.min(dim=1)
                    _, rand_idx = torch.topk(min_dist, n_dead)
                else:
                    rand_idx = torch.randperm(z_size, device=z.device)[:n_dead]

        if self.use_euclidean:
            new_embeddings = z[rand_idx].detach()
        else:
            new_embeddings = F.normalize(z[rand_idx].detach(), dim=1)

        self.embedding.weight.data[dead_codes] = new_embeddings
        if not self.use_gradient_update:
            self.ema_w.data[dead_codes] = new_embeddings * 1.0
            self.ema_cluster_size.data[dead_codes] = 1.0

    def forward(self, z):
        # K-means initialization for gradient update mode
        if self.use_gradient_update and self.training and not self._initted:
            self._init_codebook(z)

        if self.use_euclidean:
            # Euclidean distance: d = ||z - e||^2 = z^2 + e^2 - 2*z*e
            d = torch.sum(z ** 2, dim=1, keepdim=True) + \
                torch.sum(self.embedding.weight ** 2, dim=1, keepdim=True).t() - \
                2 * torch.matmul(z, self.embedding.weight.t())
            min_encoding_indices = torch.argmin(d, dim=1)
            min_encodings = F.one_hot(min_encoding_indices, self.n_e).type(z.dtype)
            z_q = self.embedding(min_encoding_indices)
        else:
            # Cosine similarity
            z_normalized = F.normalize(z, dim=1)
            embedding_normalized = F.normalize(self.embedding.weight, dim=1)
            d = torch.matmul(z_normalized, embedding_normalized.t())
            min_encoding_indices = torch.argmax(d, dim=1)
            min_encodings = F.one_hot(min_encoding_indices, self.n_e).type(z.dtype)
            z_q = torch.matmul(min_encodings, embedding_normalized)

        if self.training:
            encodings_sum = min_encodings.sum(0)
            self.ema_cluster_size.data.mul_(self.decay).add_(encodings_sum, alpha=1 - self.decay)

            n = self.ema_cluster_size.sum()
            cluster_size = (self.ema_cluster_size + self.epsilon) / (n + self.n_e * self.epsilon) * n

            if not self.use_gradient_update:
                # EMA update
                dw = torch.matmul(min_encodings.t(), z)
                self.ema_w.data.mul_(self.decay).add_(dw, alpha=1 - self.decay)
                self.embedding.weight.data.copy_(self.ema_w / cluster_size.unsqueeze(1))
                if not self.use_euclidean:
                    self.embedding.weight.data = F.normalize(self.embedding.weight.data, dim=1)
            # Gradient update: embedding.weight is updated via backward pass automatically

            if self.use_codebook_restart:
                self._restart_dead_codes(z, cluster_size)

        if self.use_gradient_update:
            # Standard VQ loss: codebook_loss + beta * commitment_loss
            commitment_loss = F.mse_loss(z_q.detach(), z)
            codebook_loss = F.mse_loss(z_q, z.detach())
            loss = codebook_loss + self.beta * commitment_loss
        else:
            # EMA mode: only commitment loss
            loss = self.beta * F.mse_loss(z_q.detach(), z)

        # Straight-through estimator
        z_q = z + (z_q - z).detach()

        e_mean = torch.mean(min_encodings, dim=0)
        perplexity = torch.exp(-torch.sum(e_mean * torch.log(e_mean + 1e-10)))

        return loss, z_q, perplexity, min_encoding_indices.unsqueeze(1)


class ResidualQuantizer(nn.Module):
    """Multi-layer Residual Quantizer."""

    def __init__(self, n_e, e_dim, beta, n_layers, quantization_dropout_rate=0.0,
                 use_shared_codebook=True, use_codebook_restart=True,
                 use_gradient_update=False, use_euclidean=False):
        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.n_layers = n_layers
        self.quantization_dropout_rate = quantization_dropout_rate
        self.use_shared_codebook = use_shared_codebook

        if use_shared_codebook:
            self.quantizer = VectorQuantizerEMA(
                n_e, e_dim, beta=beta, use_codebook_restart=use_codebook_restart,
                use_gradient_update=use_gradient_update, use_euclidean=use_euclidean)
        else:
            self.quantizers = nn.ModuleList()
            for _ in range(n_layers):
                self.quantizers.append(
                    VectorQuantizerEMA(
                        n_e, e_dim, beta=beta, use_codebook_restart=use_codebook_restart,
                        use_gradient_update=use_gradient_update, use_euclidean=use_euclidean))

    def forward(self, z):
        quantized_out = 0.0
        residual = z
        total_loss = 0.0
        all_indices = []

        if self.training and self.quantization_dropout_rate > 0:
            k = torch.randint(1, self.n_layers + 1, (1,)).item()
        else:
            k = self.n_layers

        for i in range(k):
            quantizer = self.quantizer if self.use_shared_codebook else self.quantizers[i]
            loss, z_q, perplexity, indices = quantizer(residual)

            quantized_out += z_q
            residual = residual - z_q
            total_loss += loss
            all_indices.append(indices)

        if k < self.n_layers:
            batch_size = z.shape[0]
            pad_len = self.n_layers - k
            padding = torch.zeros(batch_size, pad_len, dtype=torch.long, device=z.device)
            current_indices = torch.cat(all_indices, dim=1)
            all_indices_tensor = torch.cat([current_indices, padding], dim=1)
        else:
            all_indices_tensor = torch.cat(all_indices, dim=1)

        return total_loss, quantized_out, all_indices_tensor

    def get_codes(self, z):
        residual = z
        all_indices = []
        for i in range(self.n_layers):
            quantizer = self.quantizer if self.use_shared_codebook else self.quantizers[i]
            _, z_q, _, indices = quantizer(residual)
            residual = residual - z_q
            all_indices.append(indices)
        return torch.cat(all_indices, dim=1)

    def decode(self, codes):
        quantized_z = 0
        for i in range(self.n_layers):
            indices = codes[:, i]
            embedding_weight = self.quantizer.embedding.weight if self.use_shared_codebook else self.quantizers[
                i].embedding.weight
            z_q = F.embedding(indices, embedding_weight)
            quantized_z += z_q
        return quantized_z


class GroupedResidualQuantizer(nn.Module):
    """Grouped Residual Quantizer with multiple groups."""

    def __init__(self, n_e, e_dim, beta, n_layers, n_groups, quantization_dropout_rate=0.0,
                 use_shared_codebook=True, use_codebook_restart=True,
                 use_gradient_update=False, use_euclidean=False):
        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.n_layers = n_layers
        self.n_groups = n_groups

        assert e_dim % n_groups == 0
        self.group_dim = e_dim // n_groups

        self.groups = nn.ModuleList([
            ResidualQuantizer(n_e, self.group_dim, beta, n_layers, quantization_dropout_rate,
                              use_shared_codebook=use_shared_codebook, use_codebook_restart=use_codebook_restart,
                              use_gradient_update=use_gradient_update, use_euclidean=use_euclidean)
            for _ in range(n_groups)
        ])

    def forward(self, z):
        z_groups = torch.chunk(z, self.n_groups, dim=1)
        total_loss = 0
        quantized_z_groups = []
        all_indices_groups = []

        for i, group in enumerate(self.groups):
            loss, q_z, indices = group(z_groups[i])
            total_loss += loss
            quantized_z_groups.append(q_z)
            all_indices_groups.append(indices)

        quantized_z = torch.cat(quantized_z_groups, dim=1)
        all_indices = torch.cat(all_indices_groups, dim=1)
        return total_loss, quantized_z, all_indices

    def get_codes(self, z):
        z_groups = torch.chunk(z, self.n_groups, dim=1)
        all_indices_groups = []
        for i, group in enumerate(self.groups):
            indices = group.get_codes(z_groups[i])
            all_indices_groups.append(indices)
        return torch.cat(all_indices_groups, dim=1)

    def decode(self, codes):
        codes_groups = torch.chunk(codes, self.n_groups, dim=1)
        quantized_z_groups = []
        for i, group in enumerate(self.groups):
            q_z = group.decode(codes_groups[i])
            quantized_z_groups.append(q_z)
        return torch.cat(quantized_z_groups, dim=1)


class ResBlock(nn.Module):
    def __init__(self, dim, hidden_dim=None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = dim * 4
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
            nn.LayerNorm(dim)
        )

    def forward(self, x):
        return x + self.net(x)


class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_res_blocks=2):
        super().__init__()
        layers = [nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU()]
        for _ in range(num_res_blocks):
            layers.append(ResBlock(hidden_dim))
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class Decoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_res_blocks=2):
        super().__init__()
        layers = [nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU()]
        for _ in range(num_res_blocks):
            layers.append(ResBlock(hidden_dim))
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class GRQ(nn.Module):
    """Grouped Residual Quantizer for semantic ID generation."""

    def __init__(self, input_dim, hidden_dim, n_e, e_dim, n_layers, n_groups=1, beta=0.25,
                 quantization_dropout_rate=0.5, config=None):
        super().__init__()
        self.encoder = Encoder(input_dim, hidden_dim, e_dim)
        self.decoder = Decoder(e_dim, hidden_dim, input_dim)

        # Default values
        use_codebook_restart = True
        use_shared_codebook = True
        use_gradient_update = False
        use_euclidean = False

        if config is not None:
            use_codebook_restart = config['use_codebook_restart'] if 'use_codebook_restart' in config else True
            use_shared_codebook = config['use_shared_codebook'] if 'use_shared_codebook' in config else True
            quantization_dropout_rate = config[
                'quantization_dropout_rate'] if 'quantization_dropout_rate' in config else quantization_dropout_rate
            use_gradient_update = config['use_gradient_update'] if 'use_gradient_update' in config else False
            use_euclidean = config['use_euclidean'] if 'use_euclidean' in config else False

        if n_groups > 1:
            self.quantizer = GroupedResidualQuantizer(
                n_e, e_dim, beta, n_layers, n_groups, quantization_dropout_rate,
                use_shared_codebook=use_shared_codebook, use_codebook_restart=use_codebook_restart,
                use_gradient_update=use_gradient_update, use_euclidean=use_euclidean
            )
        else:
            self.quantizer = ResidualQuantizer(
                n_e, e_dim, beta, n_layers, quantization_dropout_rate,
                use_shared_codebook=use_shared_codebook, use_codebook_restart=use_codebook_restart,
                use_gradient_update=use_gradient_update, use_euclidean=use_euclidean
            )

    def forward(self, x):
        z = self.encoder(x)
        loss, quantized_z, indices = self.quantizer(z)
        x_recon = self.decoder(quantized_z)
        return loss, x_recon, indices

    def get_codes(self, x):
        z = self.encoder(x)
        return self.quantizer.get_codes(z)

    def decode(self, codes):
        quantized_z = self.quantizer.decode(codes)
        return self.decoder(quantized_z)
