import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

import pandas as pd

from cascadePrediction import CascadePredictor

# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------


def survival_loss(edge_prob, t_event):
    eps = 1e-8

    E, L = edge_prob.shape
    time_idx = torch.arange(L).unsqueeze(0)  # (1, L)

    event_mask = (t_event.unsqueeze(1) == time_idx)  # (E, L) bool
    before_event = (time_idx < t_event.unsqueeze(1))
    no_event = (t_event == -1)

    log_prob = torch.log(edge_prob + eps)
    log_1mprob = torch.log(1 - edge_prob + eps)

    # sum to get vector of shape (E,)
    loss_event = -(event_mask * log_prob).sum(dim=1)
    loss_survive = -(before_event * log_1mprob).sum(dim=1)
    # sum over all time steps for censored edges
    loss_censored = -(log_1mprob).sum(dim=1)
    loss = torch.where(
        no_event,
        loss_censored,
        loss_event + loss_survive
    )

    return loss.mean()

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = CascadePredictor(
        d_node=4,
        d_edge=3,
        L=3
    )

    model.train()

    # ---- Graph ----
    N = 4   # nodes
    E = 5   # edges
    L = 3   # layers

    edge_index = torch.tensor([
        [0, 0, 1, 2, 3],  # src (v)
        [1, 2, 2, 3, 0]   # dst (u)
    ])
    src, dst = edge_index

    # ---- Temporal inputs ----
    node_mask = torch.zeros(N, L, dtype=torch.bool)
    node_mask[2, 0] = True  # Node 2 is a root node
    # that has already reposted at t=0

    repost_edges = torch.zeros(E, L, dtype=torch.bool)
    repost_edges[1, 1] = True  # Node 0 reposts from node 2 before layer 1
    repost_edges[2, 1] = True  # Node 1 reposts from node 2 before layer 1
    repost_edges[4, 2] = True  # Node 3 reposts from node 0 before layer 2
    edge_ids, layer_ids = repost_edges.nonzero(as_tuple=True)

    node_mask[src[edge_ids], layer_ids] = True
    node_mask = torch.cumsum(
        node_mask,
        dim=1
    )

    edge_mask = node_mask[src]  # edge is frozen if source node
    # has reposted before layer l

    # ---- Features ----
    x = torch.randn(N, 4, dtype=torch.float64, requires_grad=True)
    edge_attr = torch.randn(E, 3, dtype=torch.float64)
    followers_count = torch.tensor([1, 1, 2, 1])  # (N,)
    influence_ratio = torch.zeros(N, L, dtype=torch.float64)  # (N, L)
    repost_counts = torch.zeros(N, L, dtype=torch.float64, device=dst.device)
    repost_counts.index_put_(
        (dst[edge_ids], layer_ids),
        torch.ones_like(edge_ids, dtype=torch.float64),
        accumulate=True
    )
    repost_counts = torch.cumsum(repost_counts, dim=1)
    influence_ratio = repost_counts / (followers_count.unsqueeze(1) + 1e-8)
    delta_t = torch.rand(E)
    t = ["2024-01-01"] * E
    timestamps = pd.to_datetime(t)
    dayofweek = torch.tensor(
        timestamps.dayofweek.values, device=repost_edges.device, dtype=torch.float64)
    hour = torch.tensor(timestamps.hour.values,
                        device=repost_edges.device, dtype=torch.float64)
    t_is_weekend = (dayofweek == 5) | (dayofweek == 6)
    t_is_afternoon = hour >= 17  # in v user's timezone
    comments = torch.rand(E)
    likes = torch.rand(E)

    edge_prob = model(
        x,
        edge_index,
        edge_attr,
        node_mask,
        edge_mask,
        followers_count,
        influence_ratio,
        delta_t,
        t_is_weekend,
        t_is_afternoon,
        comments,
        likes,
    )  # (E, L)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    optimizer.zero_grad()

    print("Edge probabilities:", edge_prob)

    t_event = torch.argmax(repost_edges.int(), dim=1)  # event happened before
    # layer, -1 if no event

    loss = survival_loss(edge_prob, t_event)

    loss.backward()

    for l in range(model.L):
        grad_norm = model.W_users[l].weight.grad.norm().item()
        print("grad W_users", model.W_users[l].weight.grad)
        print(f"W_users[{l}] grad norm:", grad_norm)
        print("grad W_edges", model.W_edges[l].weight.grad)
        print(f"W_edges[{l}] grad norm:",
              model.W_edges[l].weight.grad.norm().item())

    print("W_y grad norm:", model.W_y.weight.grad)

    for name, param in model.named_parameters():
        if param.grad is None:
            print(f"⚠️ No grad for {name}")
        elif torch.isnan(param.grad).any():
            print(f"⚠️ NaNs in grad for {name}")

    torch.autograd.gradcheck(survival_loss, (edge_prob, t_event))

    torch.autograd.gradcheck(
        lambda x, edge_attr, node_mask, edge_mask, followers_count, influence_ratio, delta_t, t_is_weekend, t_is_afternoon, comments, likes: model(
            x, edge_index, edge_attr, node_mask, edge_mask, followers_count, influence_ratio,
            delta_t, t_is_weekend, t_is_afternoon, comments, likes
        ).sum(),
        (x, edge_attr, node_mask, edge_mask, followers_count,
         influence_ratio, delta_t, t_is_weekend, t_is_afternoon, comments, likes),
        eps=1e-6,
        atol=1e-4,
        rtol=1e-3
    )
