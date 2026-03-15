import Levenshtein
import torch
import spacy

from util import *

import torch.nn.functional as F

nlp = spacy.load("en_core_web_sm")

def lemmatize(text):
    return " ".join([token.lemma_ for token in nlp(text)])

def normalized_edit_similarity(a, b, is_simple): # simple means f1-based, otherwise is accuracy
    if is_simple:
        a, b = a[0], b[0]
        dist = Levenshtein.distance(a, b)
        max_len = max(len(a), len(b))
        return 1 - dist / max_len
    else:
        a = extract_one_digit_answer(a)
        return 1.0 if a == b else 0.0

def string_f1(pred: str, gold: str, is_simple: bool) -> float:
    if not pred or not pred.strip():
        return 0.0  # silent return, no print
    pred = lemmatize(pred)
    if is_simple:
        gold = lemmatize(gold)
        return 0.1 * string_f1_simple(pred, gold) + 0.9 * edit_f1(pred, gold, is_simple)
    else:
        return normalized_edit_similarity(pred, gold, is_simple)

def string_f1_simple(pred: str, gold: str) -> float:
    """
    Very simple token-level F1 between prediction and gold answer.
    """
    import re

    def normalize(s: str) -> list[str]:
        s = s.lower().strip()
        s = re.sub(r"[^a-z0-9]+", " ", s)
        return [t for t in s.split() if t]
    if not isinstance(pred, list):
        pred = normalize(pred)
        gold = normalize(gold)
        if not gold:
            print('not gold')
            return 0.0
        if not pred:
            print('not pred')
            return 0.0

    pred_set = set(pred)
    gold_set = set(gold)
    # print(pred_set, gold_set)
    inter = pred_set & gold_set
    if not inter:
        # print('not inter')
        return 0.0

    precision = len(inter) / len(pred_set)
    recall = len(inter) / len(gold_set)
    if precision + recall == 0:
        # print('well')
        return 0.0
    return 2 * precision * recall / (precision + recall)

def edit_f1(pred: str, gold: str, is_simple: bool) -> float:
    import re

    def normalize(s: str) -> list[str]:
        s = s.lower().strip()
        s = re.sub(r"[^a-z0-9]+", " ", s)
        return [t for t in s.split() if t]

    if not isinstance(pred, list):
        pred = normalize(pred)
        gold = normalize(gold)
        if not gold:
            return 0.0
        if not pred:
            return 0.0
    return normalized_edit_similarity(pred, gold, is_simple)

def compute_hornet_rewards(qwen_outputs, ground_truth, is_simple):
    rewards = []
    for pred in qwen_outputs:
        r = string_f1(pred, ground_truth, is_simple)
        rewards.append(r)
    return rewards


def grpo_loss_v2(logits, actions, rewards, kl_logits=None, kl_coef=0.0, eps=1e-8):
    """
    logits:      [B, T]  policy logits for keep/drop
    actions:     [B, K, T]  sampled 0/1 frame selections
    rewards:     [B, K]  similarity scores in [0,1]
    kl_logits:   [B, T] optional reference logits for KL penalty
    """

    B, K, T = actions.shape # b 5 16

    # Expand logits to [B, K, T]
    keep_logits = logits.squeeze(2).unsqueeze(1).expand(-1, K, -1)

    # log(sigmoid) and log(1-sigmoid)
    log_prob_keep = -F.softplus(-keep_logits)
    log_prob_drop = -keep_logits - F.softplus(-keep_logits)

    # log-prob of each candidate
    log_probs = actions * log_prob_keep + (1 - actions) * log_prob_drop   # [B, K, T]
    log_probs = log_probs.mean(dim=2)  # [B, K]

    # -------------------------
    # 1. Advantage normalization
    # -------------------------
    # Compute per-batch baseline
    rewards = torch.tensor(rewards).to(log_probs.device)
    # print(rewards)
    baseline = rewards.mean(dim=1, keepdim=True)  # [B, 1]
    advantages = rewards - baseline               # [B, K]

    # Normalize advantages to avoid collapse
    std = advantages.std()
    if std < 1e-6:
        std = 1.0
    advantages = advantages / std

    # -------------------------
    # 2. Policy gradient term
    # -------------------------
    # print(advantages.shape, log_probs.shape) 8,4 8,5
    pg_loss = -(advantages * log_probs).mean()

    # -------------------------
    # 3. KL penalty (optional but stabilizing)
    # -------------------------
    if kl_logits is not None and kl_coef > 0:
        # KL between current and reference Bernoulli distributions
        p = torch.sigmoid(logits)
        q = torch.sigmoid(kl_logits)
        kl = p * (p/q).log() + (1-p) * ((1-p)/(1-q)).log()
        kl_loss = kl.mean()
    else:
        kl_loss = 0.0

    return pg_loss + kl_coef * kl_loss

def grpo_loss_mcq(
        logits, actions, rewards,
        kl_logits=None, kl_coef=0.0,
        ent_coef=0.01, eps=1e-8
    ):
    B, K, T = actions.shape

    # ---- 1. Extract keep logits safely ----
    keep_logits = logits[..., 0]              # [B, T]
    keep_logits = keep_logits.unsqueeze(1).expand(B, K, T)

    # ---- 2. Bernoulli log-probs ----
    log_prob_keep = -F.softplus(-keep_logits)
    log_prob_drop = -keep_logits - F.softplus(-keep_logits)

    log_probs = actions * log_prob_keep + (1 - actions) * log_prob_drop
    log_probs = log_probs.mean(dim=2)         # [B, K]

    # ---- 3. Advantage (binary rewards) ----
    rewards = torch.tensor(rewards).to(log_probs.device)
    baseline = rewards.mean(dim=1, keepdim=True)
    advantages = rewards - baseline

    # Normalize safely
    std = advantages.std()
    if std < 1e-6:
        std = 1.0
    advantages = advantages / std

    # ---- 4. Policy gradient ----
    pg_loss = -(advantages * log_probs).mean()

    # ---- 5. KL penalty ----
    if kl_logits is not None and kl_coef > 0:
        p = torch.sigmoid(logits[..., 0])
        q = torch.sigmoid(kl_logits[..., 0])
        kl = p * (p/q).log() + (1-p) * ((1-p)/(1-q)).log()
        kl_loss = kl.mean()
    else:
        kl_loss = 0.0

    # ---- 6. Entropy bonus ----
    p = torch.sigmoid(keep_logits)
    entropy = -(p * log_prob_keep + (1-p) * log_prob_drop).mean()

    return pg_loss + kl_coef * kl_loss - ent_coef * entropy