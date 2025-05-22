import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
    
class ClassificationLoss(nn.Module):
    def __init__(self, class_weights=None):
        super(ClassificationLoss, self).__init__()
        self.class_weights = class_weights
        self.loss_fn = nn.CrossEntropyLoss(weight=class_weights)  
    def forward(self, logits, targets):
        return self.loss_fn(logits, targets)

class ClusteringLoss(nn.Module):
    def forward(self, high_attention_encoding):

        if len(high_attention_encoding.size()) == 3:
            high_attention_encoding.mean(dim=-1)
        mean = torch.mean(high_attention_encoding, dim=0)
        distances = torch.sum((high_attention_encoding - mean) ** 2, dim=1)
        relative_distances = distances / torch.sum(mean ** 2)

        relative_distances = relative_distances / torch.max(relative_distances)
        
        clustering_loss = torch.mean(relative_distances)
        return clustering_loss.requires_grad_()

class DistanceLoss(nn.Module):
    def forward(self, high_attention_encoding, low_attention_encoding):
        min_len = min(high_attention_encoding.size(1), low_attention_encoding.size(1))
        high_attention_encoding = high_attention_encoding[:, :min_len, :]
        low_attention_encoding = low_attention_encoding[:, :min_len, :]

        min_batch_size = min(high_attention_encoding.size(0), low_attention_encoding.size(0))

        high_attention_encoding = high_attention_encoding[:min_batch_size, :, :]
        low_attention_encoding = low_attention_encoding[:min_batch_size, :, :]

        distances = torch.sum((high_attention_encoding - low_attention_encoding) ** 2, dim=2)
        mean_high = torch.mean(high_attention_encoding, dim=1)
        mean_low = torch.mean(low_attention_encoding, dim=1)
        relative_distances = distances / (torch.sum(mean_high ** 2, dim=1, keepdim=True) + torch.sum(mean_low ** 2, dim=1, keepdim=True))

        relative_distances = relative_distances / torch.max(relative_distances)
        
        distance_loss = torch.mean(relative_distances)
        return distance_loss.requires_grad_()


class ParetoOptimizer:
    @staticmethod
    def solve_pareto_weights(G, num_iter=10):
        lambda_ = torch.ones(3, device=G.device) / 3
        for k in range(num_iter):
            gradient = G @ lambda_
            d_candidates = torch.eye(3, device=G.device)
            losses = [gradient @ d for d in d_candidates]
            min_idx = torch.argmin(torch.stack(losses))
            d = d_candidates[min_idx]
            gamma = 2.0 / (k + 2) 
            lambda_ = (1 - gamma) * lambda_ + gamma * d
        return lambda_

class Poly1CrossEntropyLoss(nn.Module):
    def __init__(self,
                 num_classes: int,
                 epsilon: float = 0.01,
                 reduction: str = "mean",
                 weight: torch.Tensor = None):
        """
        Create instance of Poly1CrossEntropyLoss
        :param num_classes:
        :param epsilon:
        :param reduction: one of none|sum|mean, apply reduction to final loss tensor
        :param weight: manual rescaling weight for each class, passed to Cross-Entropy loss
        """
        super(Poly1CrossEntropyLoss, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.reduction = reduction
        self.weight = weight

    def forward(self, logits, labels):
        """
        Forward pass
        :param logits: tensor of shape [N, num_classes]
        :param labels: tensor of shape [N]
        :return: poly cross-entropy loss
        """
        self.weight = self.weight.to(device=logits.device) if self.weight is not None else None
        labels_onehot = F.one_hot(labels, num_classes=self.num_classes).to(device=logits.device,
                                                                           dtype=logits.dtype)
        pt = torch.sum(labels_onehot * F.softmax(logits, dim=-1), dim=-1)
        CE = F.cross_entropy(input=logits,
                             target=labels,
                             reduction='none',
                             weight=self.weight)
        poly1 = CE + self.epsilon * (1 - pt)
        if self.reduction == "mean":
            poly1 = poly1.mean()
        elif self.reduction == "sum":
            poly1 = poly1.sum()
        return poly1

class InterpretableLoss(nn.Module):
    def __init__(self, task_type='classify', class_weights=None, alpha_flag=False):
        super(InterpretableLoss, self).__init__()
        self.task_type = task_type
        if task_type == 'classify':
            self.main_loss_fn = ClassificationLoss(class_weights)

        elif task_type == 'regression':
            self.main_loss_fn = nn.MSELoss()
        else:
            raise ValueError("task_type must be 'classify' or 'regression'")

        self.clustering_loss_fn = ClusteringLoss()
        self.distance_loss_fn = DistanceLoss()
        self.alpha = 0.0
        self.beta = 0.0
        self.gamma = 0.0

        self.alpha_flag = alpha_flag

    def merge_encodings(self, encoding, mask):

        bs, seg, m, n = encoding.shape
    
        encoding_flat = encoding.view(bs * seg * n, m)
        mask_flat = mask.view(bs * seg * n)
    
        selected_encodings = encoding_flat[mask_flat]
    
        merged_encodings = selected_encodings.view(-1, m, n)
    
        return merged_encodings
    
    def forward(self, out, y, mask_flag=False):
        high_attention_output = out['high_attention_output']

        classification_loss = self.main_loss_fn(high_attention_output, y)

        regularization_loss = out.get('regularization_loss', torch.tensor(0.0).to(high_attention_output.device))

        # Scale regularization_loss to the same magnitude as classification_loss
        if regularization_loss != 0:
            regularization_loss = regularization_loss / torch.max(regularization_loss) * torch.max(classification_loss)

        if self.alpha == 0 and self.beta == 0:
            clustering_loss = torch.tensor(0.0).to(high_attention_output.device)
            distance_loss = torch.tensor(0.0).to(high_attention_output.device)
            return classification_loss, classification_loss, clustering_loss, distance_loss, regularization_loss
        
        if 'high_attention_encoding' in out and 'low_attention_encoding' in out and out['high_attention_encoding'] is not None and out['low_attention_encoding'] is not None:
            high_attention_encoding = out['high_attention_encoding']
            low_attention_encoding = out['low_attention_encoding']
            mask = out['mask']
            high_flat = self.merge_encodings(high_attention_encoding, mask)
            low_flat = self.merge_encodings(low_attention_encoding, ~mask)
            clustering_loss = self.clustering_loss_fn(high_flat) + self.clustering_loss_fn(low_flat)
            distance_loss = self.distance_loss_fn(high_flat, low_flat)
        else:
            clustering_loss = torch.tensor(0.0).to(high_attention_output.device)
            distance_loss = torch.tensor(0.0).to(high_attention_output.device)

        total_loss = classification_loss + self.alpha * distance_loss + self.beta * clustering_loss + self.gamma * regularization_loss

        # Ensure the loss has gradients
        if total_loss.grad_fn is None:
            raise RuntimeError("Loss does not have gradients. Check the computation graph.")

        return total_loss.requires_grad_(), classification_loss.requires_grad_(), distance_loss.requires_grad_(), clustering_loss.requires_grad_(), regularization_loss.requires_grad_()

    def update(self, model, optimizer, inputs, times, y, epoch, total_epochs, writter=None):
        """
        Updates alpha, beta, and gamma using Pareto Optimization.

        Args:
            model: The PyTorch model.
            optimizer: The PyTorch optimizer.
            inputs: Input data.
            y: Target data.
            epoch: Current epoch.
            total_epochs: Total number of epochs.
            writter: Optional TensorBoard writer.
        """
        model.train()

        classification_loss, distance_loss, clustering_loss, regularization_loss = self.calculate_individual_losses(model, inputs, times, y)

        if not self.alpha_flag:#  or epoch <= total_epochs // 10:
            self.alpha = 0.0
            self.beta = 0.0
            self.gamma = 0.0
            if writter is not None:
                writter.add_scalar('train/alpha', self.alpha, epoch)
                writter.add_scalar('train/beta', self.beta, epoch)
                writter.add_scalar('train/gamma', self.gamma, epoch)
            return None

        params = list(model.parameters())
        optimizer.zero_grad()
        grad1 = torch.autograd.grad(classification_loss, params, retain_graph=True, create_graph=True, allow_unused=True)
        grad2 = torch.autograd.grad(distance_loss, params, retain_graph=True, create_graph=True, allow_unused=True)
        grad3 = torch.autograd.grad(clustering_loss, params, retain_graph=True, create_graph=True, allow_unused=True)
        grad4 = torch.autograd.grad(regularization_loss, params, retain_graph=True, create_graph=True, allow_unused=True)

        # Flatten gradients and handle None values
        def flatten_grad(grad):
            return torch.cat([g.contiguous().view(-1) for g in grad if g is not None])

        g1 = flatten_grad(grad1)
        g2 = flatten_grad(grad2)
        g3 = flatten_grad(grad3)
        g4 = flatten_grad(grad4)

        # Ensure all gradients have the same size
        min_size = min(g1.size(0), g2.size(0), g3.size(0), g4.size(0))
        g1 = g1[:min_size]
        g2 = g2[:min_size]
        g3 = g3[:min_size]
        g4 = g4[:min_size]

        G = torch.zeros((4, 4), device=g1.device)
        G[0, 0] = torch.dot(g1, g1)
        G[0, 1] = G[1, 0] = torch.dot(g1, g2)
        G[0, 2] = G[2, 0] = torch.dot(g1, g3)
        G[0, 3] = G[3, 0] = torch.dot(g1, g4)
        G[1, 1] = torch.dot(g2, g2)
        G[1, 2] = G[2, 1] = torch.dot(g2, g3)
        G[1, 3] = G[3, 1] = torch.dot(g2, g4)
        G[2, 2] = torch.dot(g3, g3)
        G[2, 3] = G[3, 2] = torch.dot(g3, g4)
        G[3, 3] = torch.dot(g4, g4)

        lambda_ = ParetoOptimizer.solve_pareto_weights(G[:3,:3], num_iter=10)

        optimizer.step()
        
        self.alpha = lambda_[0].item()  
        self.beta = lambda_[1].item()   
        self.gamma = lambda_[2].item()  


        if writter is not None:
            writter.add_scalar('train/alpha', self.alpha, epoch)
            writter.add_scalar('train/beta', self.beta, epoch)
            writter.add_scalar('train/gamma', self.gamma, epoch)


    def calculate_individual_losses(self, model, inputs, times, y):
        """
        Calculates individual losses.  Separated out so we can compute grads later.

        Args:
            model: The PyTorch model.
            inputs: Input data.
            y: Target data.

        Returns:
            Tuple of (classification_loss, distance_loss, clustering_loss, regularization_loss)
        """
        model.train()
        out = model(inputs, times) # Assumes model returns a dictionary like the forward() method

        high_attention_output = out['high_attention_output']

        classification_loss = self.main_loss_fn(high_attention_output, y)
        regularization_loss = out.get('regularization_loss', torch.tensor(0.0).to(high_attention_output.device))

        if 'high_attention_encoding' in out and 'low_attention_encoding' in out and out['high_attention_encoding'] is not None and out['low_attention_encoding'] is not None:
            high_attention_encoding = out['high_attention_encoding']
            low_attention_encoding = out['low_attention_encoding']
            clustering_loss = self.clustering_loss_fn(high_attention_encoding) + self.clustering_loss_fn(low_attention_encoding)
            distance_loss = self.distance_loss_fn(high_attention_encoding, low_attention_encoding)
        else:
            clustering_loss = torch.tensor(0.0).to(high_attention_output.device)
            distance_loss = torch.tensor(0.0).to(high_attention_output.device)


        return classification_loss, distance_loss, clustering_loss, regularization_loss
