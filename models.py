import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict

class TaskEmbeddingComposer(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=128, num_heads=4, dropout=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Transformer encoder for contextual task representation
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)
        
        # Projection MLP: projects concatenated features to 512-dim embedding
        self.projection = nn.Sequential(
            nn.Linear(896, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 512)
        )
        
        self.sample_projection = nn.Linear(input_dim, 512)
        
    def forward(self, features, labels=None, return_sample_features=False):
        """
        Args:
            features: [N, D] support set features
            labels: [N] support set labels
            return_sample_features: if True, return sample-level features; otherwise, return task embedding
        Returns:
            task_embedding: [1, 512] task embedding or [N, 512] sample-level features
        """
        N, D = features.shape
        
        # Step 3: Contextual encoding via Transformer
        features_expanded = features.unsqueeze(0)  # [1, N, D]
        context_features = self.transformer(features_expanded)  # [1, N, D]
        
        if return_sample_features:
            return self.sample_projection(context_features.squeeze(0))  # [N, 512]
        
        # Step 1: Global feature statistics (mean and std)
        mu = features.mean(dim=0)  # [D]
        sigma = torch.sqrt(torch.mean((features - mu.unsqueeze(0))**2, dim=0))  # [D]
        
        # Step 2: Class prototype encoding
        if labels is not None:
            unique_labels = torch.unique(labels)
            prototypes = []
            for label in unique_labels:
                mask = labels == label
                if mask.sum() > 0:
                    proto = features[mask].mean(dim=0)  # [D]
                    prototypes.append(proto)
            if prototypes:
                prototypes = torch.stack(prototypes)  # [num_classes, D]
            else:
                prototypes = torch.zeros(0, D, device=features.device)
        else:
            # If no labels, estimate prototypes via clustering
            prototypes = self._estimate_prototypes(features)
        
        h_ctx = context_features.mean(dim=1).squeeze(0)  # [D], contextual summary
        
        # Step 4: Concatenate all components into a single vector
        prototype_dim = prototypes.shape[0] * D if len(prototypes) > 0 else D
        total_dim = D + D + prototype_dim + D
        
        # Pad or truncate to fixed dimension (896)
        if total_dim < 896:
            padding = torch.zeros(896 - total_dim, device=features.device)
            combined = torch.cat([mu, sigma, prototypes.flatten(), h_ctx, padding])
        elif total_dim > 896:
            combined = torch.cat([mu, sigma, prototypes.flatten()[:896-3*D], h_ctx])
        else:
            combined = torch.cat([mu, sigma, prototypes.flatten(), h_ctx])
        
        combined = combined[:896]  # Ensure correct dimension
        
        # Step 5: Project to final task embedding
        task_embedding = self.projection(combined.unsqueeze(0))  # [1, 512]
        
        return task_embedding
    
    def _estimate_prototypes(self, features, num_clusters=5):
        """Estimate prototypes using a simplified K-means (used when labels are not available)."""
        N, D = features.shape
        if N < num_clusters:
            num_clusters = N
        
        # Randomly initialize cluster centers
        indices = torch.randperm(N)[:num_clusters]
        centers = features[indices]
        
        # Run a few iterations of K-means
        for _ in range(3):
            dists = torch.cdist(features, centers)
            assignments = dists.argmin(dim=1)
            for k in range(num_clusters):
                mask = assignments == k
                if mask.sum() > 0:
                    centers[k] = features[mask].mean(dim=0)
        
        return centers

class ProtoLearner(nn.Module):
    def __init__(self, embedder):
        super().__init__()
        self.embedder = embedder

    def forward(self, support_feats, support_labels, query_feats):
        classes = torch.unique(support_labels)
        prototypes = []
        for cls in classes:
            mask = support_labels == cls
            if mask.sum() == 0:
                continue
            cls_feats = support_feats[mask]
            # Compute class prototype as mean of support features for this class
            proto = cls_feats.mean(0)
            prototypes.append(proto)
        prototypes = torch.stack(prototypes)
        
        # Compute squared Euclidean distance between query features and prototypes
        dists = torch.cdist(query_feats, prototypes, p=2)
        squared_dists = dists ** 2
        
        # Return negative squared distances as logits (for softmax classification)
        logits = -squared_dists
        
        return logits

class MAMLClassifier(nn.Module):
    """Standard second-order MAML few-shot classifier."""
    def __init__(self, feature_extractor, n_way=5, inner_steps=5, inner_lr=0.01):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.n_way = n_way
        self.inner_steps = inner_steps
        self.inner_lr = inner_lr
        self.classifier = nn.Linear(512, n_way)  # Assumes 512-dim features
    def forward(self, support_x, support_y, query_x):
        # Extract features for support and query sets
        support_feat = self.feature_extractor(support_x)
        query_feat = self.feature_extractor(query_x)
        # Initialize fast weights for inner-loop adaptation
        fast_weights = list(self.classifier.parameters())
        for _ in range(self.inner_steps):
            logits = F.linear(support_feat, fast_weights[0], fast_weights[1])
            loss = F.cross_entropy(logits, support_y)
            grads = torch.autograd.grad(loss, fast_weights, create_graph=True)
            fast_weights = [w - self.inner_lr * g for w, g in zip(fast_weights, grads)]
        # Use adapted weights for query prediction
        query_logits = F.linear(query_feat, fast_weights[0], fast_weights[1])
        return query_logits

class iCaRLClassifier(nn.Module):
    """Standard iCaRL implementation with memory and mean-of-exemplars classifier."""
    def __init__(self, feature_extractor, memory_size=2000, n_classes=100):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.memory_size = memory_size
        self.n_classes = n_classes
        self.memory = defaultdict(list)  # {class_idx: [features]}
    def update_memory(self, images, labels):
        features = self.feature_extractor(images).detach().cpu()
        for feat, label in zip(features, labels.cpu()):
            self.memory[int(label)].append(feat)
            # Limit number of exemplars per class
            if len(self.memory[int(label)]) > self.memory_size // self.n_classes:
                self.memory[int(label)] = self.memory[int(label)][-self.memory_size // self.n_classes:]
    def classify(self, images):
        features = self.feature_extractor(images).detach().cpu()
        class_means = {}
        for c in self.memory:
            feats = torch.stack(self.memory[c])
            class_means[c] = feats.mean(dim=0)
        preds = []
        for feat in features:
            # Assign to class with closest mean in feature space
            dists: dict = {c: torch.norm(feat - mean).item() for c, mean in class_means.items()}
            min_c = min(list(dists.keys()), key=lambda c: dists[c])
            pred = int(min_c)
            preds.append(pred)
        return torch.tensor(preds, device=images.device)
    def forward(self, support_x, support_y, query_x):
        self.update_memory(support_x, support_y)
        preds = self.classify(query_x)
        return preds

class EWCRegularizer:
    """Elastic Weight Consolidation (EWC) regularizer for continual learning."""
    
    def __init__(self, model, lambda_ewc=1.0):
        self.model = model
        self.lambda_ewc = lambda_ewc
        self.fisher_info = {}
        self.optimal_params = {}
        
    def update_fisher_info(self, dataloader, num_samples=100):
        """Update Fisher Information Matrix using current data."""
        self.model.eval()
        fisher_info = {}
        
        # Initialize Fisher info
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                fisher_info[name] = torch.zeros_like(param.data)
        
        # Compute Fisher information
        sample_count = 0
        for batch_idx, (data, target) in enumerate(dataloader):
            if sample_count >= num_samples:
                break
                
            self.model.zero_grad()
            output = self.model(data)
            loss = F.cross_entropy(output, target)
            loss.backward()
            
            for name, param in self.model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    fisher_info[name] += param.grad.data ** 2
            
            sample_count += 1
        
        # Average Fisher information
        for name in fisher_info:
            fisher_info[name] /= sample_count
        
        self.fisher_info = fisher_info
        
        # Store current optimal parameters
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.optimal_params[name] = param.data.clone()
    
    def update_fisher_info_from_features(self, features, labels):
        """Update Fisher Information Matrix using features and labels directly."""
        self.model.eval()
        fisher_info = {}
        
        # Initialize Fisher info
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                fisher_info[name] = torch.zeros_like(param.data)
        
        # Compute Fisher information using features
        self.model.zero_grad()
        logits = self.model(features)
        loss = F.cross_entropy(logits, labels)
        loss.backward()
        
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                fisher_info[name] = param.grad.data ** 2
        
        self.fisher_info = fisher_info
        
        # Store current optimal parameters
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.optimal_params[name] = param.data.clone()
    
    def compute_ewc_loss(self):
        """Compute EWC regularization loss."""
        ewc_loss = 0.0
        
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.fisher_info:
                optimal_param = self.optimal_params[name]
                fisher_info = self.fisher_info[name]
                
                ewc_loss += torch.sum(fisher_info * (param - optimal_param) ** 2)
        
        return self.lambda_ewc * ewc_loss
    
    def set_lambda(self, lambda_ewc):
        """Set EWC regularization strength."""
        self.lambda_ewc = lambda_ewc

class MetaSGDWithEWC(nn.Module):
    """MetaSGD with simplified EWC regularization for continual few-shot learning."""
    
    def __init__(self, feature_extractor, n_way=5, inner_steps=5, inner_lr=0.01, lambda_ewc=1.0):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.n_way = n_way
        self.inner_steps = inner_steps
        self.inner_lr = inner_lr
        self.lambda_ewc = lambda_ewc
        
        # Get feature dimension from feature extractor
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 84, 84)
            dummy_features = feature_extractor(dummy_input)
            feature_dim = dummy_features.shape[1]
        
        self.classifier = nn.Linear(feature_dim, n_way)
        
        # Store previous parameters for EWC
        self.previous_params = {}
        self.fisher_info = {}
        
    def update_ewc_info(self, support_x, support_y):
        """Update EWC information using current support set."""
        # Store current parameters
        for name, param in self.classifier.named_parameters():
            if param.requires_grad:
                self.previous_params[name] = param.data.clone()
        
        # Simple Fisher information approximation
        support_feat = self.feature_extractor(support_x)
        logits = self.classifier(support_feat)
        loss = F.cross_entropy(logits, support_y)
        loss.backward()
        
        # Store gradients as Fisher information approximation
        for name, param in self.classifier.named_parameters():
            if param.requires_grad and param.grad is not None:
                self.fisher_info[name] = param.grad.data ** 2
        
        # Zero gradients
        self.classifier.zero_grad()
        
    def compute_ewc_loss(self):
        """Compute simplified EWC loss."""
        ewc_loss = 0.0
        
        for name, param in self.classifier.named_parameters():
            if param.requires_grad and name in self.previous_params and name in self.fisher_info:
                prev_param = self.previous_params[name]
                fisher_info = self.fisher_info[name]
                
                ewc_loss += torch.sum(fisher_info * (param - prev_param) ** 2)
        
        return self.lambda_ewc * ewc_loss
        
    def forward(self, support_x, support_y, query_x, adaptive_params=None):
        # Extract features
        support_feat = self.feature_extractor(support_x)
        query_feat = self.feature_extractor(query_x)
        
        # Update EWC information
        self.update_ewc_info(support_x, support_y)
        
        # Use adaptive parameters if provided
        lr = adaptive_params.get('lr_scale', 1.0) * self.inner_lr if adaptive_params else self.inner_lr
        ewc_lambda = adaptive_params.get('ewc_lambda', 1.0) if adaptive_params else self.lambda_ewc
        
        # Initialize fast weights
        fast_weights = list(self.classifier.parameters())
        
        # Inner loop adaptation
        for step in range(self.inner_steps):
            logits = F.linear(support_feat, fast_weights[0], fast_weights[1])
            task_loss = F.cross_entropy(logits, support_y)
            
            # Add EWC regularization (simplified)
            ewc_loss = ewc_lambda * torch.sum((fast_weights[0] - self.previous_params.get('weight', fast_weights[0])) ** 2)
            total_loss = task_loss + ewc_loss
            
            # Compute gradients
            grads = torch.autograd.grad(total_loss, fast_weights, create_graph=True)
            fast_weights = [w - lr * g for w, g in zip(fast_weights, grads)]
        
        # Query prediction
        query_logits = F.linear(query_feat, fast_weights[0], fast_weights[1])
        return query_logits
