"""Verify ARPredictor._build_attn_mask() correctness with a small example.

Setup:
  B=1, max_lang_tokens=5, lang_lengths=3 (positions 3,4 are lang PAD)
  max_action_tokens=6, action_lengths=4 (positions 4,5 in action zone are PAD)

Full sequence (L = 5 + 3 + 1 + 6 = 15):
  pos  0: lang_0  (real)
  pos  1: lang_1  (real)
  pos  2: lang_2  (real)
  pos  3: lang_PAD
  pos  4: lang_PAD
  pos  5: z_agent (real)
  pos  6: z_hand  (real)
  pos  7: z_proprio (real)
  pos  8: BOS     (real)
  pos  9: T_1     (real)
  pos 10: T_2     (real)
  pos 11: T_3     (real)
  pos 12: T_4     (real)
  pos 13: act_PAD
  pos 14: act_PAD

Expected mask rules:
  - Lang PAD (pos 3,4): entire row = 0, entire column = 0
  - Real prefix (pos 0-2, 5-7): bidirectional among each other
  - Prefix cannot see action zone (pos 8+)
  - BOS (pos 8): sees all real prefix + itself
  - T_i (pos 9-12): sees all real prefix + BOS + T_1..T_i (causal)
  - Action PAD (pos 13,14): entire row = 0, entire column = 0
"""

import torch
from module import ARPredictor, PAD_TOKEN_ID

def main():
    # Create a minimal ARPredictor (only need mask logic, not weights).
    # proprio_dim matches the production default (9d = ee_pos(3)+quat(4)+grip(2)).
    pred = ARPredictor(
        embed_dim=32, depth=1, heads=1, dim_head=32, mlp_dim=64,
        max_action_tokens=6, max_lang_tokens=5, proprio_dim=9,
    )

    B = 1
    n_lang = 5
    lang_lengths = torch.tensor([3])  # only first 3 are real

    # Action tokens: 4 real tokens (ids 10,20,30,40) + 2 PAD
    action_tokens = torch.tensor([[10, 20, 30, 40, PAD_TOKEN_ID, PAD_TOKEN_ID]])

    L = pred.max_seq_len  # 5 + 3 + 1 + 6 = 15
    print(f"max_seq_len = {L}")

    mask = pred._build_attn_mask(n_lang, lang_lengths, action_tokens, L, torch.device("cpu"))
    mask_2d = mask[0, 0].int()  # (L, L), 1=attend, 0=blocked

    # Labels for readability
    labels = [
        "lang0", "lang1", "lang2", "lPAD3", "lPAD4",
        "z_ag", "z_hd", "z_pr",
        "BOS", "T_1", "T_2", "T_3", "T_4", "aPAD5", "aPAD6",
    ]

    # Print matrix
    header = "        " + " ".join(f"{l:>6s}" for l in labels)
    print(header)
    print("        " + "-" * (7 * len(labels)))
    for i in range(L):
        row = mask_2d[i].tolist()
        row_str = " ".join(f"{v:>6d}" for v in row)
        print(f"{labels[i]:>7s} | {row_str}")

    # ---- Automated checks ----
    errors = []

    # Check 1: Lang PAD rows (pos 3,4) should be all 0
    for p in [3, 4]:
        if mask_2d[p].any():
            errors.append(f"FAIL: lang PAD row {p} ({labels[p]}) is not all-zero")

    # Check 2: Lang PAD columns (pos 3,4) should be all 0
    for p in [3, 4]:
        if mask_2d[:, p].any():
            errors.append(f"FAIL: lang PAD column {p} ({labels[p]}) is not all-zero")

    # Check 3: Real prefix (0,1,2,5,6,7) should be bidirectional among each other
    real_prefix = [0, 1, 2, 5, 6, 7]
    for i in real_prefix:
        for j in real_prefix:
            if mask_2d[i, j] != 1:
                errors.append(f"FAIL: prefix bidir — mask[{labels[i]},{labels[j]}] = 0, expected 1")

    # Check 4: Prefix cannot see action zone (pos 8+)
    for i in real_prefix:
        for j in range(8, L):
            if mask_2d[i, j] != 0:
                errors.append(f"FAIL: prefix→action — mask[{labels[i]},{labels[j]}] = 1, expected 0")

    # Check 5: BOS (pos 8) sees all real prefix + itself
    for j in real_prefix + [8]:
        if mask_2d[8, j] != 1:
            errors.append(f"FAIL: BOS cannot see {labels[j]}")
    # BOS should NOT see future action tokens
    for j in range(9, L):
        if mask_2d[8, j] != 0:
            errors.append(f"FAIL: BOS sees future {labels[j]}")

    # Check 6: Action tokens causal — T_i sees prefix + BOS + T_1..T_i
    for i_pos in range(9, 13):  # T_1 to T_4
        # Should see all real prefix
        for j in real_prefix:
            if mask_2d[i_pos, j] != 1:
                errors.append(f"FAIL: {labels[i_pos]} cannot see prefix {labels[j]}")
        # Should see BOS + all T up to itself
        for j in range(8, i_pos + 1):
            if mask_2d[i_pos, j] != 1:
                errors.append(f"FAIL: {labels[i_pos]} cannot see {labels[j]}")
        # Should NOT see future action tokens
        for j in range(i_pos + 1, L):
            if mask_2d[i_pos, j] != 0:
                errors.append(f"FAIL: {labels[i_pos]} sees future {labels[j]}")

    # Check 7: Action PAD rows (pos 13,14) should be all 0
    for p in [13, 14]:
        if mask_2d[p].any():
            errors.append(f"FAIL: action PAD row {p} ({labels[p]}) is not all-zero")

    # Check 8: Action PAD columns (pos 13,14) should be all 0
    for p in [13, 14]:
        if mask_2d[:, p].any():
            errors.append(f"FAIL: action PAD column {p} ({labels[p]}) is not all-zero")

    # Report
    print()
    if errors:
        for e in errors:
            print(f"  {e}")
        print(f"\n{len(errors)} checks FAILED")
    else:
        print("All checks PASSED")


if __name__ == "__main__":
    main()
