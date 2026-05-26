"""
FakeNewsNet Dataset Loader for CascadePredictor
================================================
Reads the exact directory layout produced by the FakeNewsNet crawler:

    <root>/
    ├── gossipcop/
    │   ├── fake/
    │   │   ├── gossipcop-1/
    │   │   │   ├── news content.json
    │   │   │   ├── tweets/          <tweet_id>.json
    │   │   │   ├── retweets/        <tweet_id>.json   (array of retweet objects)
    │   │   │   ├── likes/           <tweet_id>.json   (array of user IDs)
    │   │   │   └── replies/         <tweet_id>.json   (reply objects)
    │   │   └── ...
    │   └── real/ ...
    ├── politifact/
    │   └── (same structure)
    ├── user_profiles/       <user_id>.json
    ├── user_timeline_tweets/ <user_id>.json
    ├── user_followers/      <user_id>.json  {"user_id":..., "followers":[...]}
    └── user_following/      <user_id>.json  {"user_id":..., "following":[...]}

Key behaviour
-------------
* STRICT USER FILTER: a cascade is included ONLY if every user involved in
  its tweets AND retweets has a file in user_profiles/.  Cascades failing
  this check are counted and reported but not added to the dataset.
* Per-source counts (politifact / gossipcop) are printed on construction so
  you can decide whether the numbers are sufficient before training.
* likes/ and replies/ folders are used to compute per-edge comment and like
  counts instead of relying on the often-missing tweet-object fields.

Node features (d_node = 14)
    0  log1p(followers_count)   — from user_profiles or follower-list length
    1  log1p(following_count)   — from user_profiles or following-list length
    2  verified                 — bool
    3  log1p(account_age_days)
    4  log1p(statuses_count)
    5  log1p(posts_last_7_days) — from user_timeline_tweets
    6  active_days_last_month/30 — from user_timeline_tweets
    7  url_ratio                — fraction of timeline tweets with URLs
    8  media_ratio              — fraction of timeline tweets with media
    9  activity_hour_sin        — circular encoding of median active hour
    10 activity_hour_cos
    11 credibility_score        — composite heuristic (see code)
    12 is_verified_page         — verified AND followers > 10 000
    13 log1p(favourites_count)  — from user_profiles

Edge features (d_edge = 4)
    0  log1p(repost_count_ij)
    1  log1p(avg_likes_ij)      — from likes/ files
    2  avg_is_weekend
    3  avg_is_afternoon

Extra per-edge tensors (inputs to CascadePredictor.forward)
    delta_t        log1p(avg seconds between consecutive reposts by same src)
    t_is_weekend   bool
    t_is_afternoon bool
    comments       log1p(avg replies per retweeted tweet, from replies/ files)
    likes          log1p(avg likes per retweeted tweet, from likes/ files)

Post features (d_post = 10)  — one vector per news article, stored as post_attr
    0  log1p(total_comments)    — sum of reply counts across all tweets
    1  log1p(total_likes)       — sum of like counts across all tweets
    2  has_images               — bool: news content.json has images
    3  has_urls                 — bool: news content.json has external URLs
    4  log1p(exclamation_count) — from article text
    5  log1p(question_count)    — from article text
    6  log1p(hashtag_count)     — from tweet texts sharing this article
    7  log1p(emoticon_count)    — from article text
    8  capital_ratio            — fraction of alpha chars that are uppercase
    9  log1p(article_length)    — char count of article body

Usage
-----
    from fakenewsnet_loader import FakeNewsNetDataset, chronological_split

    dataset = FakeNewsNetDataset(root="data/FakeNewsNet", n_layers=4)
    # → prints per-source cascade counts automatically

    train_idx, val_idx, test_idx = chronological_split(
        dataset, source="politifact"
    )
"""

import json
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from math import sqrt
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set

import numpy as np
import torch
from torch_geometric.data import Data, Dataset


# ============================================================
# Opposition scoring
# ============================================================

NEGATIVE_WORDS = [
    "fake", "false", "lie", "wrong", "not true", "untrue",
    "debunked", "don't believe", "conspiracy", "hoax",
    "scam", "fraud", "clickbait", "misleading",
]

def compute_opposition(text: Optional[str]) -> int:
    """Count how many NEGATIVE_WORDS appear in text (case-insensitive)."""
    if not text:
        return 0
    text = text.lower()
    return sum(1 for w in NEGATIVE_WORDS if w in text)


# ============================================================
# Low-level helpers
# ============================================================

def _parse_twitter_time(s: str) -> Optional[datetime]:
    if not s:
        return None
    try:
        return datetime.strptime(s, "%a %b %d %H:%M:%S %z %Y").replace(
            tzinfo=timezone.utc
        )
    except ValueError:
        return None


def _safe_load_json(path: Path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[WARNING] Failed to load JSON file: {path}")
        print(f"Reason: {repr(e)}")
        return None


def _is_weekend(dt: Optional[datetime]) -> bool:
    return bool(dt.weekday() >= 5) if dt else False


def _is_afternoon(dt: Optional[datetime]) -> bool:
    return bool(dt.hour >= 17) if dt else False


# ============================================================
# User data cache  (shared, lazy, loaded once per dataset)
# ============================================================

class UserDataCache:
    """
    Lazily loads user_profiles, user_timeline_tweets, user_followers,
    and user_following from the flat directories at <root>/.

    All four directories are siblings of politifact/ and gossipcop/.
    """

    def __init__(self, root: Path):
        self.root = root
        self._profiles:  Dict[str, dict]       = {}
        self._timelines: Dict[str, List[dict]] = {}
        self._followers: Dict[str, List[str]]  = {}
        self._following: Dict[str, List[str]]  = {}
        # pre-build sets of available user IDs for fast membership test
        self._profile_ids: Optional[set] = None

    # ------------------------------------------------------------------
    def available_profile_ids(self) -> set:
        """Set of user IDs that have a file in user_profiles/."""
        if self._profile_ids is None:
            folder = self.root / "user_profiles"
            if folder.exists():
                self._profile_ids = {p.stem for p in folder.glob("*.json")}
            else:
                self._profile_ids = set()
        return self._profile_ids

    def has_profile(self, uid: str) -> bool:
        return uid in self.available_profile_ids()

    # ------------------------------------------------------------------
    def _load(self, folder: str, uid: str):
        return _safe_load_json(self.root / folder / f"{uid}.json")

    def profile(self, uid: str) -> dict:
        if uid not in self._profiles:
            # If the user is missing, we return an empty dict so the 
            # cascade is still usable with zero feature vectors.
            d = self._load("user_profiles", uid)
            self._profiles[uid] = d if isinstance(d, dict) else {}
        return self._profiles[uid]

    def timeline(self, uid: str) -> List[dict]:
        if uid not in self._timelines:
            d = self._load("user_timeline_tweets", uid)
            self._timelines[uid] = d if isinstance(d, list) else []
        return self._timelines[uid]

    def followers(self, uid: str) -> List[str]:
        if uid not in self._followers:
            path = Path(self.root / "user_followers" / f"{uid}.json")
            if not path.exists():
                self._followers[uid] = []
                return self._followers[uid]
            d = self._load("user_followers", uid)
            # format: {"user_id": "...", "followers": [...]}
            if isinstance(d, dict):
                lst = d.get("followers", [])
            elif isinstance(d, list):
                lst = d
            else:
                lst = []
            self._followers[uid] = [str(x) for x in lst]
        return self._followers[uid]

    def following(self, uid: str) -> List[str]:
        if uid not in self._following:
            path = Path(self.root / "user_following" / f"{uid}.json")
            if not path.exists():
                self._following[uid] = []
                return self._following[uid]
            d = self._load("user_following", uid)
            # format: {"user_id": "...", "followees": [...]}
            if isinstance(d, dict):
                lst = d.get("followees", [])
            elif isinstance(d, list):
                lst = d
            else:
                lst = []
            self._following[uid] = [str(x) for x in lst]
        return self._following[uid]


# ============================================================
# Node feature extractor  →  (NODE_FEATURES_DIM,) float64 list
# ============================================================

NODE_FEATURES_DIM = 14

def _node_features(uid: str, cache: UserDataCache, now: datetime) -> List[float]:
    # If no profile exists, return a zero vector so the cascade is still usable.
    if not cache.has_profile(uid):
        return [0.0] * NODE_FEATURES_DIM
    profile = cache.profile(uid)
    profile_info = profile.get("profile_info", {}) or {}

    followers_count  = float(profile_info.get("followers_count",  0) or 0)
    following_count  = float(profile_info.get("friends_count",    0) or 0)
    verified         = float(profile_info.get("verified",    False) or False)
    notifications    = float(profile_info.get("notifications", False) or False)
    listed_count     = float(profile_info.get("listed_count",   0) or 0)
    statuses_count   = float(profile_info.get("statuses_count",   0) or 0)
    favourites_count = float(profile_info.get("favourites_count", 0) or 0)
    created_at = _parse_twitter_time(profile_info.get("created_at", ""))
    age_days   = max((now - created_at).days, 1) if created_at else 0

    # Override with explicit ID-list lengths when available
    fl = cache.followers(uid)
    fg = cache.following(uid)
    if fl:
        followers_count = float(len(fl))
    if fg:
        following_count = float(len(fg))

    # ---- Timeline features ----
    timeline  = cache.timeline(uid)
    recent_tweets = timeline.get("recent_tweets", []) or []
    week_ago  = now - timedelta(days=7)
    month_ago = now - timedelta(days=30)

    posts_week = 0
    active_days_last_month = 0
    url_c = 0
    # images are not directly available in the timeline data
    hours: List[float] = []

    for tw in recent_tweets:
        t = _parse_twitter_time(tw.get("created_at", ""))
        if t is None:
            continue
        if t >= week_ago:
            posts_week += 1
        if t >= month_ago:
            active_days_last_month += 1
        ent = tw.get("entities") or {}
        if ent.get("urls"):
            url_c += 1
        elif tw.get("retweeted_status", {}).get("entities", {}).get("urls"):
            url_c += 1
        hours.append(float(t.hour))

    n_tl        = max(len(recent_tweets), 1)
    url_ratio   = url_c   / n_tl

    if hours:
        med = float(np.median(hours))
        h_sin = float(np.sin(2 * np.pi * med / 24))
        h_cos = float(np.cos(2 * np.pi * med / 24))
    else:
        # default to 6pm
        h_sin = float(np.sin(2 * np.pi * 18 / 24))
        h_cos = float(np.cos(2 * np.pi * 18 / 24))

    ff_ratio    = followers_count / max(following_count, 1.0)
    # ideally, the credibility scores of the users would have already been computed based on
    # past activity on labeled posts. In this dataset the users do not participate in multiplle cascades, 
    # so we can only compute a heuristic credibility scre based on a combbination of user features, 
    # rendering it uncessecary to have it as a separate feature.
    # credibility = float(
    #     0.35 * verified
    #     + 0.30 * min(np.log1p(ff_ratio)       / 6.0,  1.0)
    #     + 0.20 * min(np.log1p(age_days)        / 10.0, 1.0)
    #     + 0.15 * min(np.log1p(statuses_count)  / 12.0, 1.0)
    # )

    # user is considered a "verified page" if they are verified and have 
    # more than 10k followers
    # even if the account refers to a person, they may behave like a 
    # page if they have a large following and are verified
    is_page = float(verified and followers_count > 10_000)

    return [
        np.log1p(followers_count),           # 0
        np.log1p(following_count),           # 1
        verified,                            # 2
        notifications,                       # 3
        np.log1p(listed_count),              # 4
        np.log1p(statuses_count),            # 5
        np.log1p(favourites_count),          # 6
        np.log1p(age_days),                  # 7
        np.log1p(posts_week),                # 8
        active_days_last_month / 30.0,       # 9
        url_ratio,                           # 10
        h_sin,                               # 11
        h_cos,                               # 12
        is_page,                             # 13
    ]

# ============================================================
# Post feature extractor  →  (POST_FEATURES_DIM,) float64 tensor
# ============================================================

POST_FEATURES_DIM = 9

def count_reply_and_opposition(reply_tree):

    total_replies = 0
    opposition_count = 0

    def dfs(reply):

        nonlocal total_replies, opposition_count

        total_replies += 1

        text = reply.get("text", "")
        opposition_count += compute_opposition(text)

        engagement = reply.get("engagement", {})

        child_replies = engagement.get(
            "tweet_replies",
            []
        )

        for child in child_replies:
            dfs(child)

    for r in reply_tree:
        dfs(r)

    return [total_replies, opposition_count]


def _post_features(
    self,
    news_dir: Path,
    root_id: str,
) -> torch.Tensor:
    """
    Returns a ({POST_FEATURES_DIM},) float64 tensor of each tweet of a news article.
    """

    # ── Load news content ────────────────────────────────────────
    news_article = _safe_load_json(news_dir / "news_article.json") or {}
    images  = news_article.get("images", []) or []

    root_tweet = _safe_load_json(news_dir / "tweets" / f"{root_id}.json") or {}
    text       = root_tweet.get("text", "") or ""
    root_tweet_likes = float(root_tweet.get("favorite_count", 0) or 0)
    replies = _safe_load_json(news_dir / "replies" / f"{root_id}.json")
    replies_tree = replies.get("replies", []) or []

    # ── Aggregate tweet-level counts ─────────────────────────────
    replies_count, opposition_count = count_reply_and_opposition(replies_tree)
    opposition_count = float(opposition_count)

    # ── Text-level features (from article body) ───────────────────
    has_images = float(len(images) > 0)

    # URLs: check if any href-like token exists in the text
    has_urls = float("http://" in text or "https://" in text or "www." in text)
    if not has_urls:
        print(f"[DEBUG] No URLs found in article text: {news_dir}")

    exclamation_count = float(text.count("!"))
    question_count    = float(text.count("?"))

    alpha_chars = [c for c in text if c.isalpha()]
    capital_ratio = (
        sum(1 for c in alpha_chars if c.isupper()) / len(alpha_chars)
        if alpha_chars else 0.0
    )
    article_length = float(len(text))

    # ── Hashtag count — aggregated from all sharing tweets ───────
    entities = root_tweet.get("entities") or {}
    hashtag_count = float(len(entities.get("hashtags", [])))

    return torch.tensor([
        np.log1p(replies_count),     # 0
        np.log1p(root_tweet_likes),  # 1
        opposition_count,            # 2
        has_images,                  # 3
        has_urls,                    # 4
        np.log1p(exclamation_count), # 5
        np.log1p(question_count),    # 6
        np.log1p(hashtag_count),     # 7
        capital_ratio,               # 8
        np.log1p(article_length),    # 9
    ], dtype=torch.float64)

# ============================================================
# Per-cascade graph builder
# ============================================================

LABEL_MAP = {"fake": 1, "real": 0}
SOURCES   = ["politifact", "gossipcop"]

EdgeKey = Tuple[str, str]

class _GraphBuilder:

    def __init__(self, n_layers: int, user_cache: UserDataCache, source: str, delta_t: float = 1.0):
        self.L     = n_layers
        self.user_cache = user_cache
        self.root = self.user_cache.root
        self.source = source
        self.delta_t = delta_t
        self.now = datetime.now(tz=timezone.utc)
        self.users_per_cascade, self.edges = self._build_social_network_interactions()
        # self.compress_cascades_keys = self._compress_cascades()

    # ------------------------------------------------------------------

    def _build_social_network_interactions(self) -> Tuple[Dict[str, set[str]], Dict[EdgeKey, dict]]:
        """
        Builds social network based on following, followers and  edge_index and edge_interactions tensors from retweet events.
        Returns:
            edge_features    : dict mapping (src_uid, dst_uid) to dict of edge features
        """
        network_path = self.root / f"{self.source}_network.json"
        if network_path.exists():
            with open(network_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            self.users_per_cascade = {
                k: set(v)
                for k, v in data["users_per_cascade"].items()
            }
            edges_raw = data["edges"]

            self.edges = {
                tuple(k.strip("()").split(",")): v
                for k, v in edges_raw.items()
            }
            valid = True

            if not self.users_per_cascade:
                print(f"[WARNING] users_per_cascade is empty in {network_path}, rebuilding from scratch.")
                valid = False
            
            if not self.edges:
                print(f"[WARNING] edges is empty in {network_path}, rebuilding from scratch.")
                valid = False

            if valid:
                return self.users_per_cascade, self.edges

        news_dirs = []
        for label in LABEL_MAP:
            source_dir = self.root / f"{self.source}_{label}"
            news_dirs += list(source_dir.glob("*"))

        edges = defaultdict(lambda: {"retweet_count": 0, "like_count": 0, "reply_count": 0, "opposition_score": 0.0, "hours_since_last_retweet": 0.0})
        users_per_cascade = defaultdict(set)
        for news_dir in news_dirs:
            tweets_dir   = news_dir / "tweets"
            for tweet in tweets_dir.glob("*.json"):
                tweet_id = str(tweet.stem)
                d = _safe_load_json(tweet) or {}
                user = d.get("user", {})
                dst = user.get("id_str", str(user.get("id", "")))
                users_per_cascade[tweet_id].add(dst)

                # If user is not in user_profiles, we add them
                if not self.user_cache.has_profile(dst):
                    save_path = Path(self.root / "user_profiles") / f"{dst}.json"

                    with open(save_path, "w", encoding="utf-8") as f:
                        json.dump(user, f, ensure_ascii=False, indent=2)
                    
                # Update edges given the retweet events for this tweet 
                retweets_file = news_dir / "retweets" / f"{tweet_id}.json"
                retweets = _safe_load_json(retweets_file) or {}
                for rt in retweets.get("retweets", []):
                    src_user = rt.get("user", {})
                    src = src_user.get("id_str", str(src_user.get("id", "")))
                    if src:
                        users_per_cascade[tweet_id].add(src)

                        # If user is not in user_profiles, we add them
                        if not self.user_cache.has_profile(src):
                            save_path = Path(self.root / "user_profiles") / f"{src}.json"

                            with open(save_path, "w", encoding="utf-8") as f:
                                json.dump(user, f, ensure_ascii=False, indent=2)

                        edge_key = (src, dst)
                        edges[edge_key]["retweet_count"] += 1
                        rt_time = _parse_twitter_time(rt.get("created_at", ""))
                        if rt_time:
                            hours_since_last_rt = (self.now - rt_time).total_seconds() / 3600
                        else:
                            hours_since_last_rt = 1.0
                        if edges[edge_key]["hours_since_last_retweet"] == 0.0:
                            edges[edge_key]["hours_since_last_retweet"] = hours_since_last_rt
                        else:
                            edges[edge_key]["hours_since_last_retweet"] = min(edges[edge_key]["hours_since_last_retweet"], hours_since_last_rt)
                
                # Update edges given the like events for this tweet
                likes_file = news_dir / "likes" / f"{tweet_id}.json"
                likes = _safe_load_json(likes_file) or {}
                for like_user_id in likes.get("likes", []):
                    src = str(like_user_id)
                    users_per_cascade[tweet_id].add(src)
                    edge_key = (src, dst)
                    edges[edge_key]["like_count"] += 1

                # Update edges given the reply events for this tweet
                replies_file = news_dir / "replies" / f"{tweet_id}.json"
                replies = _safe_load_json(replies_file) or {}
                for reply in replies.get("replies", []):
                    src = str(reply.get("user_id", "") or reply.get("user", ""))
                    if src:
                        users_per_cascade[tweet_id].add(src)

                         # If user is not in user_profiles, we add them
                        if not self.user_cache.has_profile(src):
                            save_path = Path(self.root / "user_profiles") / f"{src}.json"

                            with open(save_path, "w", encoding="utf-8") as f:
                                json.dump(user, f, ensure_ascii=False, indent=2)

                        edge_key = (src, dst)
                        edges[edge_key]["reply_count"] += 1
                        edges[edge_key]["opposition_score"] += compute_opposition(reply.get("text", ""))
        
        data = {
            "users_per_cascade": {
                k: list(v)
                for k, v in users_per_cascade.items()
            },
            "edges": {
                f"({u},{v})": val
                for (u, v), val in edges.items()
            }
        }

        with open(network_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

        return users_per_cascade, edges
    
    def _compress_cascades(self) -> List[Set[str]]:
        """
            Returns a dict mapping a list of cascade_ids whose users form a unified graph 
        """
        # Compress cascades that share users into one, 
        # to capture more edges in the social network
        compress_cascades_keys : List[Set[str]] = []

        cascades_left = set(self.users_per_cascade.keys())

        for current_cascade_id in cascades_left:
            users_of_this_cascade = self.users_per_cascade.get(current_cascade_id, set())
            for cascade_id, users in self.users_per_cascade.items():
                if cascade_id == current_cascade_id:
                    continue
                for user in users:
                    if user in users_of_this_cascade:
                        # Compress cascades
                        for key in compress_cascades_keys:
                            if cascade_id in key or current_cascade_id in key:
                                key.update([cascade_id, current_cascade_id])
                                break
                            else:
                                new_key = set([cascade_id, current_cascade_id])
                                compress_cascades_keys.append(new_key)
                                break
                        cascades_left.discard(cascade_id)
                        break
                    
        return compress_cascades_keys

    # ------------------------------------------------------------------
    def _collect_edges_of_cascade(self, cascade_id: str) -> Tuple[Dict[EdgeKey, dict], Set[str]]:
        """
        Returns:
            edges_of_cascade: list of edge dicts with key (src_uid, dst_uid) that belong to this cascade,
            along with their features (retweet_count, like_count, reply_count, opposition_score).
        """

        edges_of_cascade: Dict[EdgeKey, dict] = {}
        
        # For each user in the cascade, also include the following edges from the social network, 
        # to capture more interactions between users in the cascade
        users_of_cascade_expanded = set(self.users_per_cascade.get(cascade_id, set()))
        user_followers_dir = self.root / "user_followers"
        user_following_dir = self.root / "user_following"
        for user in self.users_per_cascade.get(cascade_id, set()):
            user_followers_path = user_followers_dir / f"{user}.json"
            # else check if user is available in the cache
            if user_followers_path.exists():
                followers_data = _safe_load_json(user_followers_path) or {}
                followers = followers_data.get("followers", [])
                for src in followers:
                    edge_key = (str(src), user)
                    # if (src, dst) does not exist in edges, default to zero edge features
                    if edge_key not in self.edges:
                        edges_of_cascade[edge_key] = {
                            "retweet_count": 0,
                            "like_count": 0,
                            "reply_count": 0,
                            "opposition_score": 0.0,
                            "hours_since_last_retweet": 0.0,
                        }
                    users_of_cascade_expanded.update(str(src))

            user_following_path = user_following_dir / f"{user}.json"
            # else check if user is available in the cache
            if user_following_path.exists():
                following_data = _safe_load_json(user_following_path) or {}
                following = following_data.get("followees", []) or []
                for dst in following:
                    edge_key = (user, str(dst))
                    if edge_key not in self.edges:
                        edges_of_cascade[edge_key] = {
                            "retweet_count": 0,
                            "like_count": 0,
                            "reply_count": 0,
                            "opposition_score": 0.0,
                            "hours_since_last_retweet": 0.0,
                        }
                    users_of_cascade_expanded.update(str(dst))

        # Add the edges from the retweet, like and reply events for this cascade
        for edge_key, features in self.edges.items():
            src, dst = edge_key
            if src in users_of_cascade_expanded and dst in users_of_cascade_expanded:
                edges_of_cascade[edge_key] = features

        return edges_of_cascade, users_of_cascade_expanded

    # ------------------------------------------------------------------
    def enough_users_present(self, users_of_cascade_expanded: Set[str],
                              min_coverage: float) -> bool:
        """
        Returns True if at least `min_coverage` fraction of users in this
        cascade have a profile in user_profiles/.  Users without a profile
        receive a zero feature vector at build time.
        """

        if not users_of_cascade_expanded:
            return False
        n_present = sum(1 for uid in users_of_cascade_expanded if self.user_cache.has_profile(uid))
        return (n_present / len(users_of_cascade_expanded)) >= min_coverage
    
    # ------------------------------------------------------------------
    def build(self, cascade: Path, label: int) -> Optional[Data]:
        cascade_id = cascade.stem
        edges_of_cascade, users_of_cascade_expanded = self._collect_edges_of_cascade(cascade_id)


        news_dir = cascade.parent.parent
        retweets_dir = news_dir / "retweets"
        retweets_data = _safe_load_json(retweets_dir / f"{cascade_id}.json") or {}
        retweet_events = retweets_data.get("retweets", [])

        if len(retweet_events) == 0:
            return None
        
        # Ordered user list → node indices
        # uid2idx  = {uid: i for i, uid in enumerate(users_of_cascade_expanded)}
        N        = len(users_of_cascade_expanded)

        # ---- Node features (N, {NODE_FEATURES_DIM}) ----
        x = torch.tensor(
            [_node_features(uid, self.cache, self.now) for uid in users_of_cascade_expanded],
            dtype=torch.float64,
        )

        node_index = {uid: i for i, uid in enumerate(users_of_cascade_expanded)}

        # ---- Edge features (E, {EDGE_FEATURES_DIM}) and edge_index (2, E) ----
        edge_attr = torch.tensor(
            [
                [np.log1p(edge_features["retweet_count"]),
                np.log1p(edge_features["like_count"]),
                float(edge_features["opposition_score"] > 0.0),
                float(edge_features["reply_count"] > 0),
                float(edge_features["hours_since_last_retweet"])] for edge_features in edges_of_cascade.values()
            ],
            dtype=torch.float64,
        )
        edge_index = torch.tensor(
            [[node_index[src], node_index[dst]] for (src, dst) in edges_of_cascade.keys()],
            dtype=torch.long,
        )

        # ---- Post features (d_post = {POST_FEATURES_DIM}) ----
        post_attr = _post_features(self, cascade, cascade_id)

        # ---- Build masks ----
        edge_mask = torch.zeros(edge_index.size(1), dtype=torch.bool)
        node_mask = torch.zeros(x.size(0), dtype=torch.bool)

        dst_user = cascade.get("user", {})
        dst = dst_user.get("id_str", str(dst_user.get("id", "")))
        dst_node_idx = node_index[dst]
        tweet_time = _parse_twitter_time(cascade.get("created_at", ""))

        E = len(edges_of_cascade)
        t_is_afternoon_list = np.zeros(E, dtype=bool)
        t_is_weekend_list = np.zeros(E, dtype=bool)
        comments_list = np.zeros(E)
        likes_list = np.zeros(E)

        # for all edges with this dst, set the temporal features of the edges
        edges_index_dst = torch.where(edge_index[1] == dst_node_idx)[0].tolist()

        t_is_afternoon_list[edges_index_dst] = _is_afternoon(tweet_time)
        t_is_weekend_list[edges_index_dst] = _is_weekend(tweet_time)
        comments_list[edges_index_dst] = post_attr[0].item()
        likes_list[edges_index_dst] = post_attr[1].item()

        # influence of root user
        retweet_counts = np.zeros(self.L)
        influence_ratio = torch.zeros(N, dtype=torch.float64)

        for rt in retweet_events:
            src_user = rt.get("user", {})
            src = src_user.get("id_str", str(src_user.get("id", "")))
            src_node_idx = node_index[src]

            rt_time = _parse_twitter_time(rt.get("created_at", ""))
            dt = (rt_time - tweet_time).total_seconds() / 3600 if rt_time and tweet_time else self.delta_t
            layer_mask = min(int(dt // self.delta_t), self.L - 1)

            edge_mask[torch.where((edge_index[0] == src_node_idx))[0].item(), layer_mask] = True
            node_mask[src_node_idx, layer_mask] = True

            # for all edges with this src as their dst, set the temporal features of the edges
            edges_index_src = torch.where(edge_index[1] == src_node_idx)[0].tolist()
            
            t_is_afternoon_list[edges_index_src] = (rt_time.hour >= 17)
            t_is_weekend_list[edges_index_src] = (rt_time.weekday() >= 5)
            # get retweets count as comments count is not directly available in the data, 
            # and get favorite_count as likes count for this retweet event
            comments_list[edges_index_src] = rt.get("retweet_count", 0)
            likes_list[edges_index_src] = rt.get("favorite_count", 0)

            retweet_counts[dst_node_idx] += 1
            influence_ratio[src_node_idx] = rt.get("retweet_count", 0) / (x[src_node_idx, 0].item() + 1e-8)

        influence_ratio[dst_node_idx] = retweet_counts / (x[dst_node_idx, 0].item() + 1e-8)

        edge_src, edge_dst = edge_index[0], edge_index[1]
        edge_index     = torch.tensor([edge_src, edge_dst], dtype=torch.long)
        delta_t        = torch.tensor(delta_t,         dtype=torch.float64)
        t_is_weekend   = torch.tensor(t_is_weekend_list,    dtype=torch.bool)
        t_is_afternoon = torch.tensor(t_is_afternoon_list,  dtype=torch.bool)
        comments       = torch.tensor(comments_list,        dtype=torch.float64)
        likes          = torch.tensor(likes_list,           dtype=torch.float64)

        followers_count      = x[:, 0]

        # ── Debug: print diagnostics for the first cascade checked ──────
        if not hasattr(self, '_debug_printed'):
            self._debug_printed = True
            print(f"\n[DEBUG] First cascade: {cascade_id}")
            print(f"  edges found     : {edges_of_cascade.items()}")
            print(f"  unique users     : {users_of_cascade_expanded}")
            print(f"  profile_ids count: {len(self.cache.available_profile_ids())}")
            print(f"  post features     : {_post_features(self, cascade, self.root)}")
            print(f"  node features of first 5 users:")
            for i, uid in enumerate(users_of_cascade_expanded):
                if i >= 5:
                    break
                print(f"    {uid}: {_node_features(uid, self.cache)}")
            print()

        return Data(
            x               = x,
            edge_index      = edge_index,
            edge_attr       = edge_attr,
            post_attr       = post_attr,
            y               = torch.tensor([label], dtype=torch.long),
            node_mask       = node_mask,
            edge_mask       = edge_mask,
            followers_count = followers_count,
            influence_ratio = influence_ratio,
            delta_t         = delta_t,
            t_is_weekend    = t_is_weekend,
            t_is_afternoon  = t_is_afternoon,
            comments        = comments,
            likes           = likes,
        )

    # ------------------------------------------------------------------
    # def _temporal_masks(self, N, E, edge_src, edge_timestamps):
    #     L     = self.L
    #     valid = [ts for ts in edge_timestamps if ts is not None]
    #     if len(valid) < 2:
    #         return torch.zeros(N, L, dtype=torch.bool), torch.zeros(E, L, dtype=torch.bool)
    #     t_min = min(valid)
    #     span  = max((max(valid) - t_min).total_seconds(), 1.0)
    #     nm    = torch.zeros(N, L, dtype=torch.bool)
    #     em    = torch.zeros(E, L, dtype=torch.bool)
    #     for l in range(L):
    #         cutoff = span * (l + 1) / L
    #         for e_idx, (src_i, ts) in enumerate(zip(edge_src, edge_timestamps)):
    #             if ts and (ts - t_min).total_seconds() <= cutoff:
    #                 nm[src_i, l] = True
    #                 em[e_idx,  l] = True
    #     return nm, em

    # def _influence_ratio(self, N, edge_index, followers_count, node_mask):
    #     L  = self.L
    #     ir = torch.zeros(N, L, dtype=torch.float64)
    #     for l in range(L):
    #         rc = torch.zeros(N, dtype=torch.float64)
    #         for i, (src_i, dst_i) in enumerate(edge_index.t().tolist()):
    #             if dst_i == dst:
    #                 edges_index_dst.append(i)
    #         for si, di in zip(edge_src, edge_dst):
    #             if node_mask[si, l]:
    #                 rc[di] += 1.0
    #         ir[:, l] = rc / (followers_count.double() + 1e-8)
    #     return ir


# ============================================================
# Dataset
# ============================================================

class FakeNewsNetDataset(Dataset):
    """
    PyG Dataset for FakeNewsNet with strict user-profile filtering.

    On construction it scans all cascades, checks that every participating
    user has a profile, and prints a summary table like:

        ┌─────────────────┬──────────┬──────────┬─────────┬──────────────┐
        │ Source          │   fake   │   real   │  total  │  excluded    │
        ├─────────────────┼──────────┼──────────┼─────────┼──────────────┤
        │ politifact      │     180  │     190  │    370  │  430 (53.7%) │
        │ gossipcop       │    2100  │    2200  │   4300  │  600 (12.2%) │
        └─────────────────┴──────────┴──────────┴─────────┴──────────────┘

    Each Data item carries a `source` attribute ("politifact"/"gossipcop")
    and a `news_id` attribute for traceability.

    Args:
        root       : path that contains politifact/, gossipcop/,
                     user_profiles/, user_timeline_tweets/, etc.
        sources    : which sub-datasets to include
        n_layers   : GNN depth L (temporal mask resolution)
        transform  : optional PyG transform
        verbose    : print the summary table (default True)
    """

    def __init__(
        self,
        root: str,
        source: str,
        n_layers: int = 4,
        min_coverage: float = 0.0,
        transform=None,
        verbose: bool = True,
    ):
        super().__init__(root=None, transform=transform)
        self._root        = Path(root)
        self.source       = source
        self.L            = n_layers
        self.min_coverage = min_coverage

        cache         = UserDataCache(self._root)
        self._builder = _GraphBuilder(n_layers=n_layers, user_cache=cache, source=source)

        self._index: List[dict] = []   # list of {path, label, source, news_id}
        fake_dir = self._root / f"{self.source}_fake"
        real_dir = self._root / f"{self.source}_real"
        fake_folders = sorted([p for p in fake_dir.iterdir() if p.is_dir()])
        real_folders = sorted([p for p in real_dir.iterdir() if p.is_dir()])
        folders = fake_folders + real_folders
        self.cascades = []
        for folder in folders:
            # iterate through tweets in the folder and store paths
            for cascade in (folder / "tweets").glob("*.json"):
                self.cascades.append(cascade)

        # self._build_index(verbose)

    # ------------------------------------------------------------------
    def _build_index(self, verbose: bool):
        """
        Scan every news directory.  Include cascade if at least
        `min_coverage` fraction of its users have a profile (missing
        users receive zero feature vectors).  Collect counts for the
        summary table.
        """
        # counts[source][label_str] = {"included": int, "excluded": int}
        counts: Dict[str, Dict[str, Dict[str, int]]] = {
            self.source: {lbl: {"included": 0, "excluded": 0} for lbl in LABEL_MAP}
        }

        for label_str, label_int in LABEL_MAP.items():
            base = self._root / f"{self.source}_{label_str}"
            if not base.exists():
                continue
            for news_dir in sorted(base.iterdir()):
                if not news_dir.is_dir():
                    continue
                for cascade_dir in news_dir.glob("tweets/*.json"):
                    edges_of_cascade, users_of_cascade_expanded = self._builder._collect_edges_of_cascade(
                        cascade_dir.stem
                    )
                    if self._builder.enough_users_present(users_of_cascade_expanded, self.min_coverage):
                        self._index.append({
                            "path":     news_dir,
                            "label":    label_int,
                            "source":   self.source,
                            "news_id":  news_dir.name,
                        })
                        counts[self.source][label_str]["included"] += 1
                    else:

                        counts[self.source][label_str]["excluded"] += 1

        if verbose:
            self._print_summary(counts)

    # ------------------------------------------------------------------
    @staticmethod
    def _print_summary(counts):
        print("\nFakeNewsNet cascade counts (strict user-profile filter)")
        print("─" * 66)
        print(f"{'Source':<16} {'fake':>6} {'real':>6} {'total':>7}  {'excluded':>14}")
        print("─" * 66)
        grand_inc = grand_exc = 0
        for src, lbls in counts.items():
            inc_f = lbls["fake"]["included"]
            inc_r = lbls["real"]["included"]
            exc   = lbls["fake"]["excluded"] + lbls["real"]["excluded"]
            total = inc_f + inc_r
            total_all = total + exc
            pct   = 100.0 * exc / max(total_all, 1)
            print(f"{src:<16} {inc_f:>6} {inc_r:>6} {total:>7}  {exc:>6} ({pct:4.1f}%)")
            grand_inc += total
            grand_exc += exc
        print("─" * 66)
        grand_all = grand_inc + grand_exc
        grand_pct = 100.0 * grand_exc / max(grand_all, 1)
        print(f"{'TOTAL':<16} {'':>6} {'':>6} {grand_inc:>7}  {grand_exc:>6} ({grand_pct:4.1f}%)")
        print("─" * 66)
        print()

    # ------------------------------------------------------------------
    def len(self) -> int:
        return len(self._index)

    def get(self, idx: int) -> Optional[Data]:
        label_dir = self.cascades[idx].parent.parent.parent
        label = LABEL_MAP[label_dir.name.split("_")[-1]]
        print(f"\n[DEBUG] Building graph for index {idx}  source={self.source}  news_id={self.cascades[idx].parent.parent.name}  label={label}")
        g = self._builder.build(self.cascades[idx], label)
        if g is not None:
            g.source  = self.source
            g.news_id = self.cascades[idx].parent.parent.name
        return g

    # ------------------------------------------------------------------
    def indices_for(self, source: str) -> List[int]:
        """Return all dataset indices belonging to `source`."""
        return [i for i, e in enumerate(self._index) if e["source"] == source]

    def label_for(self, idx: int) -> int:
        return self._index[idx]["label"]


# ============================================================
# Chronological split  (per source or across all)
# ============================================================

def chronological_split(
    dataset: FakeNewsNetDataset,
    source: Optional[str] = None,
    train_ratio: float = 0.7,
    val_ratio:   float = 0.1,
) -> Tuple[List[int], List[int], List[int]]:
    """
    Return (train_idx, val_idx, test_idx) for either one source or all.

    The dataset index preserves sorted directory order, which approximates
    chronological order.  Pass source="politifact" or source="gossipcop"
    to split within one source only.
    """
    idxs = dataset.indices_for(source) if source else list(range(len(dataset)))
    n    = len(idxs)
    n_tr = int(n * train_ratio)
    n_va = int(n * val_ratio)
    return (
        idxs[:n_tr],
        idxs[n_tr: n_tr + n_va],
        idxs[n_tr + n_va:],
    )


# ============================================================
# Collate helper
# ============================================================

def collate_fn(batch):
    """Drop None graphs (cascades that failed to build at get() time)."""
    return [g for g in batch if g is not None]


# ============================================================
# Smoke test
# ============================================================

if __name__ == "__main__":
    import sys

    root = sys.argv[1] if len(sys.argv) > 1 else "data/FakeNewsNet"

    dataset = FakeNewsNetDataset(root=root, source="gossipcop", n_layers=4, min_coverage=0.0)

    # print(f"Total cascades in dataset : {len(dataset)}")
    # print(f"  politifact              : {len(dataset.indices_for('politifact'))}")
    # print(f"  gossipcop               : {len(dataset.indices_for('gossipcop'))}")

    # Show one sample graph
    # for i in range(min(30, len(dataset))):
    i = 0
    g = dataset.get(i)
    if g is not None:
        print(f"\nSample [{i}]  source={g.source}  news_id={g.news_id}"
                f"  label={'fake' if g.y.item() else 'real'}")
        print(f"  x              : {tuple(g.x.shape)}")
        print(f"  edge_index     : {tuple(g.edge_index.shape)}")
        print(f"  edge_attr      : {tuple(g.edge_attr.shape)}")
        print(f"  node_mask      : {tuple(g.node_mask.shape)}")
        print(f"  edge_mask      : {tuple(g.edge_mask.shape)}")
        print(f"  influence_ratio: {tuple(g.influence_ratio.shape)}")
        print(f"  likes[:4]      : {g.likes[:4].tolist()}")
        print(f"  comments[:4]   : {g.comments[:4].tolist()}")
        # break

    # Per-source splits
    for src in ["politifact", "gossipcop"]:
        tr, va, te = chronological_split(dataset, source=src)
        print(f"\n{src}  →  train={len(tr)}  val={len(va)}  test={len(te)}")