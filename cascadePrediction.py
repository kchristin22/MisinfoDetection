import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.utils import softmax

# ---------------------------------------------------------------------------
# Visibility weight function  w(t, Δt)
# ---------------------------------------------------------------------------


class VisibilityWeight(nn.Module):
    """
    Computes w(t, Δt) for each neighbour u of v.

      - If u hasn't reposted yet   -> w = 1
      - If u has reposted          -> w = gamma*exp(-gamma*dt)
                                         + w_weekend(t)
                                         + w_afternoon(t)
                                         + w_endorsement

    All scalar parameters are learnable. gamma, lambda_weekend, and
    kappa_afternoon are stored as raw unconstrained values and passed
    through softplus in forward() to keep them strictly positive.

    Args:
        gamma_init:           initial value for the exponential decay rate (γ)
        lambda_weekend_init:  initial value for the weekend bonus (λ)
        kappa_afternoon_init: initial value for the afternoon bonus (κ)
    """

    def __init__(
        self,
        gamma_init: float = 1.0,
        lambda_weekend_init: float = 0.1,
        kappa_afternoon_init: float = 0.1,
        w_comments_init: float = 1.0,
        w_likes_init: float = 1.0,
    ):
        super().__init__()

        # Store raw (unconstrained) parameters; softplus ensures positivity
        # at runtime so the exponential decay and additive bonuses stay valid.
        self.w_gamma = nn.Parameter(
            torch.tensor(gamma_init, dtype=torch.float64))
        self.w_lambda_weekend = nn.Parameter(
            torch.tensor(lambda_weekend_init, dtype=torch.float64))
        self.w_kappa_afternoon = nn.Parameter(
            torch.tensor(kappa_afternoon_init, dtype=torch.float64))

        # Endorsement weights — unconstrained, sign can be meaningful
        self.w_comments = nn.Parameter(torch.tensor(
            w_comments_init, dtype=torch.float64))
        self.w_likes = nn.Parameter(torch.tensor(
            w_likes_init, dtype=torch.float64))

    # Positive-constrained accessors
    @property
    def gamma(self) -> Tensor:
        return F.softplus(self.w_gamma)

    @property
    def lambda_weekend(self) -> Tensor:
        return F.softplus(self.w_lambda_weekend)

    @property
    def kappa_afternoon(self) -> Tensor:
        return F.softplus(self.w_kappa_afternoon)

    def forward(
        self,
        repost_edges: Tensor,    # bool  (E,) – has neighbour u reposted?
        delta_t: Tensor,         # float (E,) – time since u reposted
                                 # (0 if not yet)
        t_is_weekend: Tensor,    # bool  (E,) – is t a weekend day?
        t_is_afternoon: Tensor,  # bool  (E,) – is t in the afternoon
                                 # (>=5pm in user's timezone)?
        comments: Tensor,        # float (E,) – comment count for u
        likes: Tensor,           # float (E,) – like count for u
    ) -> Tensor:
        """Returns visibility weights of shape (E,)."""

        gamma = self.gamma
        exp_decay = gamma * torch.exp(-gamma * delta_t)
        w_weekend = self.lambda_weekend * t_is_weekend.double()
        w_afternoon = self.kappa_afternoon * t_is_afternoon.double()
        w_endorse = torch.sigmoid(
            self.w_comments * comments + self.w_likes * likes)

        w_reposted = exp_decay + w_weekend + w_afternoon + w_endorse

        return torch.where(repost_edges.bool(), w_reposted, torch.ones_like(w_reposted))


class CascadePredictor(nn.Module):
    def __init__(self, d_node: int, d_edge: int, L: int, sigma=F.relu):
        super().__init__()

        self.L = L
        self.sigma = sigma

        # Learnable weights
        self.w_homophily = nn.Parameter(torch.tensor(1, dtype=torch.float64))
        self.w_followers = nn.Parameter(torch.tensor(1, dtype=torch.float64))
        self.w_influence = nn.Parameter(torch.tensor(1, dtype=torch.float64))

        # Message passing layers
        self.W_users = nn.ModuleList([
            nn.Linear(2 * d_node, d_node, dtype=torch.float64) for _ in range(L)
        ])

        self.W_edges = nn.ModuleList([
            nn.Linear(d_edge, d_edge, dtype=torch.float64) for _ in range(L)
        ])

        # Final classifier
        self.W_y = nn.Linear(d_edge + d_node, 1, dtype=torch.float64)

        # Visibility weight function
        self.visibility = VisibilityWeight()

    def forward(
        self,
        x,                  # (N, d_node)
        edge_index,         # (2, E)
        edge_attr,          # (E, d_edge)
        node_mask,          # (N, L) bool: 1 = frozen node
                            # (has reposted before layer l)
        edge_mask,          # (E, L) bool: 1 = edge frozen
                            # (source node has reposted before layer l)
        followers_count,    # (N,)
        influence_ratio,    # (N, L)
        delta_t,            # (E,)
        t_is_weekend,       # (E,)
        t_is_afternoon,     # (E,)
        comments,           # (E,)
        likes,              # (E,)
    ):
        N = x.size(0)
        E = edge_index.size(1)

        h = x
        h_e = edge_attr

        src, dst = edge_index  # v -> u
        h_v = h[src]  # (E, d)
        h_u = h[dst]
        h_v_dot_h_u = (h_v * h_u)

        node_mask_l = node_mask[:, 0]
        edge_mask_l = edge_mask[:, 0]
        influence_ratio_l = influence_ratio[:, 0]

        probs = []

        for l in range(self.L):

            # ---- Attention scores a_vu ----
            homophily = h_v_dot_h_u.sum(dim=-1)  # dot product

            attn_logits = (
                self.w_homophily * homophily +
                self.w_followers * followers_count[dst] +
                node_mask_l[dst] * self.w_influence * influence_ratio_l[dst]
            )

            # group softmax (per source node)
            a_vu = softmax(attn_logits, src)

            # ---- Visibility weights ----
            w_vis = self.visibility(
                edge_mask_l, delta_t, t_is_weekend, t_is_afternoon, comments, likes
            )  # (E,)

            # ---- Message aggregation (for src nodes that haven't reposted)----
            messages = (
                (~edge_mask_l).unsqueeze(-1)
                * w_vis.unsqueeze(-1)
                * a_vu.unsqueeze(-1)
                * h_u
            )  # (E, d)

            agg = torch.zeros_like(
                h, dtype=torch.float64).index_add(0, src, messages)

            # ---- Node update ----
            h_new = self.W_users[l](
                torch.cat([agg, h], dim=-1)
            )
            h_new = torch.sigmoid(h_new)

            # apply mask (freeze reposted nodes)
            v_mask_node = node_mask_l.unsqueeze(-1)
            h = v_mask_node * h + (~v_mask_node) * h_new

            # normalize
            h = F.normalize(h, p=2, dim=-1)

            # ---- Edge update ----
            h_e_new = torch.sigmoid(self.W_edges[l](h_e))

            # freeze edges where repost happened
            e_mask_edge = edge_mask_l.unsqueeze(-1)
            h_e = e_mask_edge * h_e + (~e_mask_edge) * h_e_new

            h_e = F.normalize(h_e, p=2, dim=-1)

            # ---- Edge representation ----
            h_v = h[src]
            h_u = h[dst]
            h_v_dot_h_u = (h_v * h_u)

            h_vu = torch.cat([
                h_e,
                h_v_dot_h_u
            ], dim=-1)

            # ---- Prediction ----
            probs_l = torch.sigmoid(self.W_y(h_vu)).squeeze(-1)  # (E,)
            probs.append(probs_l)

            # ---- Update mask ----
            if l < (self.L - 1):
                if self.training:
                    node_mask_l = node_mask[:, l + 1]
                    edge_mask_l = edge_mask[:, l + 1]
                    influence_ratio_l = influence_ratio[:, l + 1]
                else:
                    with torch.no_grad():
                        predicted_edges = probs_l > 0.5
                        new_nodes = torch.zeros_like(node_mask_l)
                        new_nodes[src[predicted_edges]] = True
                        node_mask_l |= new_nodes
                        edge_mask_l |= predicted_edges

                        repost_counts = torch.zeros(
                            N,
                            device=x.device,
                            dtype=torch.float64
                        )
                        repost_counts.index_add_(
                            0,
                            dst,
                            predicted_edges.double()
                        )
                        influence_ratio_l = (
                            influence_ratio_l * (followers_count + 1e-8) + repost_counts) / (followers_count + 1e-8)

        edge_prob = torch.stack(probs, dim=1)  # (E, L)

        return edge_prob
