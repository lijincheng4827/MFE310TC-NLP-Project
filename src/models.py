"""Neural network definitions: FocalLoss and DistilBERT SentimentClassifier."""
import torch
from torch import nn
from transformers import AutoModel


class FocalLoss(nn.Module):
    """Multi-class focal loss to put more weight on hard-to-classify examples."""
    def __init__(self, gamma=1.5):
        super().__init__()
        self.gamma = gamma

    def forward(self, logits, labels):
        logp = logits.log_softmax(-1).gather(1, labels[:, None]).squeeze(1)
        return (-((1 - logp.exp()) ** self.gamma) * logp).mean()


class SentimentClassifier(nn.Module):
    """DistilBERT classifier supporting standard [CLS] pooling and contrast-aware residual pooling."""
    def __init__(self, model_name, improved=False, encoder=None):
        super().__init__()
        self.encoder = encoder if encoder is not None else AutoModel.from_pretrained(model_name)
        self.improved = improved
        width = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(0.2)
        self.classifier = nn.Linear(width, 3)
        if improved:
            # Extra layers for token attention pooling and residual projection
            self.attn_proj = nn.Linear(width, 1)
            self.res_proj = nn.Linear(width, width)
            nn.init.normal_(self.res_proj.weight, std=0.02)
            nn.init.zeros_(self.res_proj.bias)

    def forward(self, input_ids, attention_mask, content_mask=None,
                contrast_mask=None, return_attention=False):
        """Forward pass returning class logits and optional token attention weights."""
        states = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        cls_vec = states[:, 0]
        weights = torch.zeros_like(attention_mask, dtype=states.dtype)
        fused = cls_vec
        if self.improved:
            content = content_mask.bool()
            # Fallback to index 0 if content mask is empty
            safe = content.clone()
            safe[~safe.any(1), 0] = True
            scores = self.attn_proj(states).squeeze(-1).masked_fill(~safe, -torch.inf)
            weights = scores.softmax(-1)
            global_vec = (states * weights.unsqueeze(-1)).sum(1)
            # Pool tokens after the final contrast word (or last 25% tail if no contrast word)
            count = content.sum(1, keepdim=True)
            tail = content & (content.cumsum(1) > (count * 0.75).floor())
            marked = contrast_mask.bool() & content
            has_contrast = marked.any(1, keepdim=True)
            pool = torch.where(has_contrast, marked, tail)
            contrast_vec = (states * pool.unsqueeze(-1)).sum(1) / pool.sum(1, keepdim=True).clamp_min(1)
            # For long reviews (>40 tokens) with a contrast marker, weight the post-contrast clause at 75%
            gate = (has_contrast & (count > 40)).to(states.dtype)
            fused = (1 - 0.75 * gate) * cls_vec + 0.75 * gate * contrast_vec + 0.15 * self.res_proj(global_vec)
        logits = self.classifier(self.dropout(fused))
        return (logits, weights) if return_attention else logits
