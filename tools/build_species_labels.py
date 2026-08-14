"""Build the bird label-embedding file from TreeOfLife-200M precomputed text embeddings.

Filters the full TOL matrix to class Aves and saves models/species_labels.npz with the
embedding rows plus scientific/common/family names and the model's logit scale. Runtime
classification is then softmax(logit_scale * image_emb @ emb.T), no text encoder needed.

Usage: uv run python tools/build_species_labels.py [--out models/species_labels.npz]
"""

import argparse
import json
from pathlib import Path

import numpy as np
import open_clip
import torch
from huggingface_hub import hf_hub_download

TOL_REPO = "imageomics/TreeOfLife-200M"
MODEL_STR = "hf-hub:imageomics/bioclip-2"
CLASS_RANK_IDX = 2
FAMILY_RANK_IDX = 4

# TOL common-name casing is inconsistent ("Great blue heron"), normalize to bird-name
# title case, connector words stay lowercase ("Greater Bird-of-paradise" style)
LOWERCASE_WORDS = {"of", "the", "and"}


def title_case(name: str) -> str:
    """Capitalize each space/hyphen-separated word except connectors."""

    def cap(word: str) -> str:
        return word if word in LOWERCASE_WORDS else word[:1].upper() + word[1:]

    return " ".join("-".join(cap(p) for p in w.split("-")) for w in name.lower().split(" "))


def build(out: Path) -> None:
    names_path = hf_hub_download(TOL_REPO, "embeddings/txt_emb_species.json", repo_type="dataset")
    emb_path = hf_hub_download(TOL_REPO, "embeddings/txt_emb_species.npy", repo_type="dataset")
    with open(names_path) as f:
        names = json.load(f)
    emb = np.load(emb_path, mmap_mode="r")
    assert emb.shape[0] == len(names) or emb.shape[1] == len(names)
    if emb.shape[1] == len(names):
        emb = emb.T

    rows, scientific, common, family = [], [], [], []
    for i, (ranks, common_name) in enumerate(names):
        if len(ranks) == 7 and ranks[CLASS_RANK_IDX] == "Aves":
            rows.append(i)
            scientific.append(f"{ranks[-2]} {ranks[-1]}")
            common.append(title_case(common_name) if common_name else "")
            family.append(ranks[FAMILY_RANK_IDX])
    named = sum(bool(c) for c in common)
    print(f"{len(rows)} Aves rows of {len(names)} | with common name: {named}")

    model, _ = open_clip.create_model_from_pretrained(MODEL_STR)
    with torch.no_grad():
        logit_scale = float(model.logit_scale.exp())
    print(f"logit scale {logit_scale:.2f}")

    np.savez_compressed(
        out,
        emb=np.ascontiguousarray(emb[rows]).astype(np.float32),
        scientific=np.array(scientific),
        common=np.array(common),
        family=np.array(family),
        logit_scale=np.float32(logit_scale),
    )
    print(f"saved {out} ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="models/species_labels.npz")
    args = parser.parse_args()
    build(Path(args.out))
