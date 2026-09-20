"""Small behavior-conditioned sequential contrastive models, not an eSASRec reproduction."""
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

KINDS = ('clicks', 'carts', 'orders')
NEURAL_FEATURES = ['neural_score', 'neural_rank', 'neural_item_known']


def encode_events(events, vocab, length=30):
    """Right padded; only visible events affect item/type/relative-age features."""
    events = events[-length:]
    if not events:
        raise ValueError('An observed prefix is required')
    items = np.zeros(length, np.int64)
    types = np.zeros(length, np.int64)
    ages = np.zeros(length, np.int64)
    now = events[-1]['ts']
    for i, e in enumerate(events):
        items[i] = vocab.get(e['aid'], 1)  # 0=padding, 1=unknown, real ids start at 2
        types[i] = KINDS.index(e['type']) + 1
        ages[i] = min(31, 1 + int(math.log2(1 + max(0, now-e['ts']) / 60000)))
    return items, types, ages


def negative_mask(positive_sets, pool):
    """Exclude ALL known same-task future positives, not merely the sampled target."""
    pool = np.asarray(pool)
    return np.asarray([np.isin(pool, values) for values in positive_sets])


def task_balanced_loss(pos, neg, mask, kinds, temperature=.12):
    logits = torch.cat([pos[:, None], neg.masked_fill(mask, -1e4)], dim=1) / temperature
    losses = -F.log_softmax(logits, dim=1)[:, 0]
    means = [losses[kinds == k].mean() for k in range(3) if (kinds == k).any()]
    return torch.stack(means).mean()


class BehaviorEncoder(nn.Module):
    def __init__(self, nitems, encoder='transformer', dim=64, length=30):
        super().__init__()
        self.encoder = encoder
        self.item = nn.Embedding(nitems, dim, padding_idx=0)
        self.behavior = nn.Embedding(4, dim, padding_idx=0)
        self.age = nn.Embedding(32, dim, padding_idx=0)
        self.position = nn.Embedding(length, dim)
        self.input_norm = nn.LayerNorm(dim)
        self.heads = nn.ModuleList([nn.Linear(dim, dim, bias=False) for _ in KINDS])
        if encoder == 'transformer':
            layer = nn.TransformerEncoderLayer(dim, 2, dim*2, .1, batch_first=True, norm_first=True)
            self.blocks = nn.TransformerEncoder(layer, 2, enable_nested_tensor=False)
        elif encoder != 'pool':
            raise ValueError(encoder)
        self.register_buffer('causal', torch.triu(torch.ones(length, length, dtype=torch.bool), diagonal=1))
        nn.init.normal_(self.item.weight, std=.03)
        for emb in (self.behavior, self.age, self.position):
            nn.init.normal_(emb.weight, std=.01)
        if encoder == 'transformer':
            for block in self.blocks.layers:
                nn.init.xavier_uniform_(block.self_attn.in_proj_weight)
                for m in block.modules():
                    if isinstance(m, nn.Linear):
                        nn.init.xavier_uniform_(m.weight)
                        if m.bias is not None: nn.init.zeros_(m.bias)
        for head in self.heads: nn.init.eye_(head.weight)
        with torch.no_grad():
            self.item.weight[0].zero_(); self.behavior.weight[0].zero_(); self.age.weight[0].zero_()

    def encode(self, items, types, ages):
        mask = items == 0
        lengths = (~mask).sum(1)
        x = self.item(items) + self.behavior(types) + self.age(ages)
        if self.encoder == 'transformer':
            x = self.input_norm(x + self.position(torch.arange(items.shape[1], device=items.device)))
            x = self.blocks(x, mask=self.causal[:items.shape[1], :items.shape[1]], src_key_padding_mask=mask)
            h = x[torch.arange(len(x), device=x.device), lengths-1]
        else:
            distance = lengths[:, None]-1-torch.arange(items.shape[1], device=items.device)
            w = (.8 ** distance.clamp(min=0)).masked_fill(mask, 0)
            h = self.input_norm((x*w[:, :, None]).sum(1)/w.sum(1)[:, None])
        return torch.stack([F.normalize(head(h), dim=-1) for head in self.heads], dim=1)

    def vectors(self, items):
        return F.normalize(self.item(items), dim=-1)

    def candidate_scores(self, encoded, kinds, candidates):
        query = encoded[torch.arange(len(encoded), device=encoded.device), kinds]
        score = (query[:, None, :] * self.vectors(candidates)).sum(-1)
        return score.masked_fill(candidates < 2, -1.01)
