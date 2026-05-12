import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from cascadePrediction import CascadePredictor

# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def survival_loss(edge_prob, t_event):
    eps = 1e-8 

    E, L = edge_prob.shape
    time_idx = torch.arange(L).unsqueeze(0) # (1, L)

    event_mask = (t_event.unsqueeze(1) == time_idx) # (E, L) bool
    before_event = (time_idx < t_event.unsqueeze(1))
    no_event = (t_event == -1)

    log_prob = torch.log(edge_prob + eps)
    log_1mprob = torch.log(1 - edge_prob + eps)

    loss_event = -(event_mask * log_prob).sum(dim=1) # sum to get vector of shape (E,)
    loss_survive = -(before_event * log_1mprob).sum(dim=1)
    loss_censored = -(log_1mprob).sum(dim=1) # sum over all time steps for censored edges
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
    L=2
    )

    model.train()

    # ---- Graph ----
    N = 4   # nodes
    E = 5   # edges
    L = 2   # layers

    edge_index = torch.tensor([
        [0, 0, 1, 2, 3],  # src (v)
        [1, 2, 2, 3, 0]   # dst (u)
    ])

    # ---- Features ----
    x = torch.randn(N, 4, dtype=torch.float64, requires_grad=True)
    edge_attr = torch.randn(E, 3, dtype=torch.float64)

    followers_count = torch.rand(N)
    influence_ratio = torch.rand(N)

    # ---- Temporal inputs ----
    has_reposted = torch.zeros(E, L, dtype=torch.bool)
    has_reposted[0, 1] = True
    has_reposted[2, 1] = True

    node_mask = torch.zeros(N, L, dtype=torch.bool)
    node_mask[1, 1] = True
    node_mask[2, 1] = True
    delta_t = torch.rand(E)
    t = ["2024-01-01"] * E
    comments = torch.rand(E)
    likes = torch.rand(E)

    t_event = torch.argmax(has_reposted.int(), dim=1)
    t_event[has_reposted.sum(dim=1) == 0] = -1  # censored

    edge_prob = model(
        x,
        edge_index,
        edge_attr,
        followers_count,
        influence_ratio,
        has_reposted,
        delta_t,
        t,
        comments,
        likes,
        node_mask
    )  # (E, L)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    optimizer.zero_grad()

    print("Edge probabilities:", edge_prob)

    loss = survival_loss(edge_prob, t_event)

    loss.backward()

    for l in range(model.L):
        grad_norm = model.W_users[l].weight.grad.norm().item()
        print(f"W_users[{l}] grad norm:", grad_norm)

    print("W_y grad norm:", model.W_y.weight.grad)

    for name, param in model.named_parameters():
        if param.grad is None:
            print(f"⚠️ No grad for {name}")
        elif torch.isnan(param.grad).any():
            print(f"⚠️ NaNs in grad for {name}")

    torch.autograd.gradcheck(survival_loss, (edge_prob, t_event))

    torch.autograd.gradcheck(
        lambda x, edge_attr, followers_count, influence_ratio, delta_t, comments, likes: model(
            x, edge_index, edge_attr, followers_count, influence_ratio,
            has_reposted, delta_t, t, comments, likes, node_mask
        ).sum(),
        (x, edge_attr, followers_count, influence_ratio, delta_t, comments, likes),
        eps=1e-6,
        atol=1e-4,
        rtol=1e-3
    )