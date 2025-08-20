import torch
import torch.nn.functional as F


class TripletLoss(torch.nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        distance_positive = F.pairwise_distance(anchor, positive, 2)
        distance_negative = F.pairwise_distance(anchor, negative, 2)
        losses = torch.relu(distance_positive - distance_negative + self.margin)
        return losses.mean()

class CMAELoss(torch.nn.Module):
    def __init__(self, margin_rr=1.0, margin_nn=1.0, margin_rn=1.0, margin_nr=1.0,
                 lamda_rr=1.0, lamda_nn=1.0, lamda_rn=0.001, lamda_nr=0.001, lamda_mae=1.0):
        super(CMAELoss, self).__init__()
        self.margin_rr = margin_rr
        self.margin_nn = margin_nn
        self.margin_rn = margin_rn
        self.margin_nr = margin_nr
        self.lamda_rr = lamda_rr
        self.lamda_nn = lamda_nn
        self.lamda_rn = lamda_rn
        self.lamda_nr = lamda_nr

        self.lamda_mae = lamda_mae
    
    def forward(self, mae_loss, rtl_emb, net_emb, lamda_rr=None, lamda_nn=None, lambda_cs=None, lamda_mae=None):
        mae_loss_rtl = mae_loss
        rtl_ori, rtl_pos, rtl_neg = rtl_emb
        net_ori, net_pos, net_neg = net_emb
        
        if lamda_rr:
            self.lamda_rr = lamda_rr
        if lamda_nn:
            self.lamda_nn = lamda_nn
        if lambda_cs:
            self.lamda_rn = self.lamda_nr = lambda_cs
        if lamda_mae:
            self.lamda_mae = lamda_mae
            
        loss_rtl = self.triplet_loss(rtl_ori, rtl_pos, rtl_neg, self.margin_rr)
        loss_net = self.triplet_loss(net_ori, net_pos, net_neg, self.margin_nn)
        loss_rn = self.triplet_loss(rtl_ori, net_pos, net_neg, self.margin_rn)
        loss_nr = self.triplet_loss(net_ori, rtl_pos, rtl_neg, self.margin_nr)

        loss_joint = self.lamda_rr * loss_rtl + self.lamda_nn * loss_net +\
                     self.lamda_rn * loss_rn + self.lamda_nr * loss_nr +\
                     0.2*(self.lamda_mae * mae_loss_rtl)
        
        return loss_joint

    def triplet_loss(self, anchor, positive, negative, margin):
        distance_positive = F.pairwise_distance(anchor, positive, 2)
        distance_negative = F.pairwise_distance(anchor, negative, 2)
        losses = torch.relu(distance_positive - distance_negative + margin)
        return losses.mean()
    


__all__ = ['InfoNCE', 'info_nce']


class InfoNCE(torch.nn.Module):
    """
    Calculates the InfoNCE loss for self-supervised learning.
    This contrastive loss enforces the embeddings of similar (positive) samples to be close
        and those of different (negative) samples to be distant.
    A query embedding is compared with one positive key and with one or more negative keys.

    References:
        https://arxiv.org/abs/1807.03748v2
        https://arxiv.org/abs/2010.05113

    Args:
        temperature: Logits are divided by temperature before calculating the cross entropy.
        reduction: Reduction method applied to the output.
            Value must be one of ['none', 'sum', 'mean'].
            See torch.nn.functional.cross_entropy for more details about each option.
        negative_mode: Determines how the (optional) negative_keys are handled.
            Value must be one of ['paired', 'unpaired'].
            If 'paired', then each query sample is paired with a number of negative keys.
            Comparable to a triplet loss, but with multiple negatives per sample.
            If 'unpaired', then the set of negative keys are all unrelated to any positive key.

    Input shape:
        query: (N, D) Tensor with query samples (e.g. embeddings of the input).
        positive_key: (N, D) Tensor with positive samples (e.g. embeddings of augmented input).
        negative_keys (optional): Tensor with negative samples (e.g. embeddings of other inputs)
            If negative_mode = 'paired', then negative_keys is a (N, M, D) Tensor.
            If negative_mode = 'unpaired', then negative_keys is a (M, D) Tensor.
            If None, then the negative keys for a sample are the positive keys for the other samples.

    Returns:
         Value of the InfoNCE Loss.

     Examples:
        >>> loss = InfoNCE()
        >>> batch_size, num_negative, embedding_size = 32, 48, 128
        >>> query = torch.randn(batch_size, embedding_size)
        >>> positive_key = torch.randn(batch_size, embedding_size)
        >>> negative_keys = torch.randn(num_negative, embedding_size)
        >>> output = loss(query, positive_key, negative_keys)
    """

    def __init__(self, temperature=0.1, reduction='mean', negative_mode='unpaired'):
        super().__init__()
        self.temperature = temperature
        self.reduction = reduction
        self.negative_mode = negative_mode

    def forward(self, query, positive_key, negative_keys=None):
        return info_nce(query, positive_key, negative_keys,
                        temperature=self.temperature,
                        reduction=self.reduction,
                        negative_mode=self.negative_mode)


def info_nce(query, positive_key, negative_keys=None, temperature=0.1, reduction='mean', negative_mode='unpaired'):
    # Check input dimensionality.
    if query.dim() != 2:
        raise ValueError('<query> must have 2 dimensions.')
    if positive_key.dim() != 2:
        raise ValueError('<positive_key> must have 2 dimensions.')
    if negative_keys is not None:
        if negative_mode == 'unpaired' and negative_keys.dim() != 2:
            raise ValueError("<negative_keys> must have 2 dimensions if <negative_mode> == 'unpaired'.")
        if negative_mode == 'paired' and negative_keys.dim() != 3:
            raise ValueError("<negative_keys> must have 3 dimensions if <negative_mode> == 'paired'.")

    # Check matching number of samples.
    if len(query) != len(positive_key):
        raise ValueError('<query> and <positive_key> must must have the same number of samples.')
    if negative_keys is not None:
        if negative_mode == 'paired' and len(query) != len(negative_keys):
            raise ValueError("If negative_mode == 'paired', then <negative_keys> must have the same number of samples as <query>.")

    # Embedding vectors should have same number of components.
    if query.shape[-1] != positive_key.shape[-1]:
        raise ValueError('Vectors of <query> and <positive_key> should have the same number of components.')
    if negative_keys is not None:
        if query.shape[-1] != negative_keys.shape[-1]:
            raise ValueError('Vectors of <query> and <negative_keys> should have the same number of components.')

    # Normalize to unit vectors
    query, positive_key, negative_keys = normalize(query, positive_key, negative_keys)
    if negative_keys is not None:
        # Explicit negative keys

        # Cosine between positive pairs
        positive_logit = torch.sum(query * positive_key, dim=1, keepdim=True)

        if negative_mode == 'unpaired':
            # Cosine between all query-negative combinations
            negative_logits = query @ transpose(negative_keys)

        elif negative_mode == 'paired':
            query = query.unsqueeze(1)
            negative_logits = query @ transpose(negative_keys)
            negative_logits = negative_logits.squeeze(1)

        # First index in last dimension are the positive samples
        logits = torch.cat([positive_logit, negative_logits], dim=1)
        labels = torch.zeros(len(logits), dtype=torch.long, device=query.device)
    else:
        # Negative keys are implicitly off-diagonal positive keys.

        # Cosine between all combinations
        logits = query @ transpose(positive_key)

        # Positive keys are the entries on the diagonal
        labels = torch.arange(len(query), device=query.device)

    return F.cross_entropy(logits / temperature, labels, reduction=reduction)


def transpose(x):
    return x.transpose(-2, -1)


def normalize(*xs):
    return [None if x is None else F.normalize(x, dim=-1) for x in xs]