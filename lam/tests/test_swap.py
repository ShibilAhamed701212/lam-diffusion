"""CPU check of the attribute-swap consistency pieces (no model weights needed)."""
import random
from types import SimpleNamespace

import torch

from lam.descriptor import TEXT_FIELDS, collate, swap_text_field
from lam.encoder import FIELD_SEQ, N_IDENTITY, TOKENS_PER_CHAR
from lam.losses import swap_consistency


def test_swap_changes_only_one_colour_field():
    d = {"global": {}, "characters": [{"gender": "female", "eye_color": "blue eyes", "hair_color": "red hair",
                                       "outfit": "dress"}]}
    batch = collate([d])
    cf, f = swap_text_field(batch, random.Random(0))
    assert TEXT_FIELDS[f] in ("eye_color", "hair_color")
    for g, name in enumerate(TEXT_FIELDS):
        same = cf.texts[0][0][g] == batch.texts[0][0][g]
        assert same != (g == f), name  # exactly the swapped field differs
    assert batch.texts[0][0][TEXT_FIELDS.index("outfit")] == "dress"  # original not mutated


def test_swap_consistency_ignores_swapped_field_only():
    ref = SimpleNamespace(char_tokens=torch.randn(1, 1, TOKENS_PER_CHAR, 8),
                          char_token_mask=torch.ones(1, 1, TOKENS_PER_CHAR, dtype=torch.bool))
    f = TEXT_FIELDS.index("eye_color")
    a, b = N_IDENTITY + f * FIELD_SEQ, N_IDENTITY + (f + 1) * FIELD_SEQ
    cf = SimpleNamespace(char_tokens=ref.char_tokens.clone(), char_token_mask=ref.char_token_mask)
    cf.char_tokens[:, :, a:b] += 5.0   # change inside the swapped field: allowed
    assert swap_consistency(ref, cf, f).item() == 0.0
    cf.char_tokens[:, :, b] += 1.0     # leak into the next field: penalised
    assert swap_consistency(ref, cf, f).item() > 0.0
