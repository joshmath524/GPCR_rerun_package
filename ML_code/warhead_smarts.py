"""
Curated covalent warhead SMARTS for scaffold±warhead evals.

Priority order is most-specific → least-specific. When assigning exclusive
binary features, the first matching pattern wins (avoids maleimide also
counting as acrylamide / α,β-unsaturated carbonyl, etc.).

Broad/noisy motifs (plain sulfonamide, sulfone, sulfoxide, nitrile,
generic carbamate, etc.) are intentionally omitted.
"""

from __future__ import annotations

# (column_suffix, SMARTS) — order = priority for exclusive assignment
CURATED_WARHEAD_SMARTS_PRIORITY: list[tuple[str, str]] = [
    ("maleimide", r"[#6]1=[#6]-[#6](=[#8])-[#7]-[#6]-1=[#8]"),
    ("vinyl_sulfonamide", r"[#6]=[#6]-[SX4](=[#8])(=[#8])-[#7]"),
    ("vinyl_sulfone", r"[#6]=[#6]-[SX4](=[#8])(=[#8])"),
    ("iodoacetamide", r"[#7]-[#6](=[#8])-[CH2]I"),
    ("chloroacetamide", r"[#7]-[#6](=[#8])-[CH2]Cl"),
    ("bromoacetamide", r"[#7]-[#6](=[#8])-[CH2]Br"),
    ("fluoroacetamide", r"[#7]-[#6](=[#8])-[CH2]F"),
    ("haloethyl_carbamate", r"[#7]-[#6](=[#8])-[#8]-[#6]-[#6]-[Cl,Br,I]"),
    ("acrylamide", r"[#6]=[#6]-[#6](=[#8])-[#7]"),
    ("michael_acrylate", r"[#6]=[#6]-[#6](=[#8])-[#8]"),
    ("alpha_halo_amide", r"[#7]-[#6](=[#8])-[#6]-[F,Cl,Br,I]"),
    ("alpha_halo_ketone", r"[#6]C(=O)C[F,Cl,Br,I]"),
    ("epoxide", r"[#6]1-[#6]-[#8]-1"),
    ("disulfide", r"[#16X2]-[#16X2]"),
    ("boronic_acid", r"[BX3]([OX2H,OX1-])[OX2H,OX1-]"),
    ("sulfonyl_bicyclobutane", r"[#8]=[#16](-[#6]12-[#6]-[#6]-1-[#6]-2)=[#8]"),
    ("alpha_beta_unsaturated_carbonyl", r"[#6]=[#6]-[#6]=[#8]"),
]

# Back-compat name used by older scripts (full 27 count set no longer primary)
WARHEAD_SMARTS: dict[str, str] = {
    f"warhead_{name}": smarts for name, smarts in CURATED_WARHEAD_SMARTS_PRIORITY
}
WARHEAD_COLUMNS = list(WARHEAD_SMARTS.keys())

MIN_SUPPORT = 50  # drop binary cols with fewer than this many positives
BINARY_PREFIX = "WHB_"
HAS_WARHEAD_COL = "has_covalent_warhead"
