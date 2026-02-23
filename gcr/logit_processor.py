# =========================
# Trie and LogitProcessor
# =========================
import torch
from transformers import LogitsProcessor
from gcr.trie import ProcessTrie

class TrieConstrainedLogitsProcessor(LogitsProcessor):
    """
    HuggingFace-native hook that masks scores so only trie-allowed next tokens remain.
    """
    def __init__(self, trie: ProcessTrie):
        super().__init__()
        self.trie = trie

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        # input_ids: [batch, seq_len], scores: [batch, vocab_size]
        batch, vocab = scores.shape
        device = scores.device

        for b in range(batch):
            prefix = input_ids[b].tolist()
            allowed = self.trie.allowed_next(prefix)

            if not allowed:
                # If nothing is allowed, discourage any continuation (end generation soon)
                scores[b, :] = float("-inf")
                continue

            # Keep only allowed token logits, set others to -inf
            allowed_idx = torch.tensor(list(allowed), device=device, dtype=torch.long)
            # Guard in case some ids fall outside current vocab size
            allowed_idx = allowed_idx[(allowed_idx >= 0) & (allowed_idx < vocab)]
            mask = torch.full((vocab,), float("-inf"), device=device)
            if allowed_idx.numel() > 0:
                mask[allowed_idx] = scores[b, allowed_idx]
            scores[b] = mask

        return scores