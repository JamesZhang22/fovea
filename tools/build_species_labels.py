"""Build the bird label-embedding file from TreeOfLife-200M precomputed text embeddings.

Filters the full TOL matrix to class Aves and saves models/species_labels.npz with the
embedding rows plus scientific/common/family names, IOC breeding-range codes for the
optional region filter, and the model's logit scale. Runtime classification is then
softmax(logit_scale * image_emb @ emb.T), no text encoder needed.

Usage: uv run python tools/build_species_labels.py [--out models/species_labels.npz]
"""

import argparse
import json
import re
import urllib.request
from pathlib import Path

import numpy as np
import open_clip
import openpyxl
import torch
from huggingface_hub import hf_hub_download

TOL_REPO = "imageomics/TreeOfLife-200M"
MODEL_STR = "hf-hub:imageomics/bioclip-2"
CLASS_RANK_IDX = 2
FAMILY_RANK_IDX = 4

# IOC World Bird List (CC-BY 4.0), breeding-range codes drive the region filter
IOC_URL = "https://worldbirdnames.org/master_ioc_list_v15.2.xlsx"
IOC_CODES = {
    "NA", "MA", "SA", "EU", "AF", "OR", "AU", "AN", "AO", "PO", "IO", "TrO", "TO", "NO", "SO",
}  # fmt: skip


def ioc_ranges(cache_dir: Path) -> dict[str, str]:
    """Scientific name -> comma-joined IOC breeding-range codes, parsed from the master list."""
    xlsx = cache_dir / "ioc_master_list.xlsx"
    if not xlsx.exists():
        urllib.request.urlretrieve(IOC_URL, xlsx)
    ws = openpyxl.load_workbook(xlsx, read_only=True).active
    ranges: dict[str, str] = {}
    genus = ""
    for row in ws.iter_rows(min_row=5, values_only=True):
        if row[5]:
            genus = str(row[5]).strip()
        epithet, subspecies, breeding = row[6], row[7], row[11]
        if not epithet or subspecies:
            continue
        codes = [t for t in re.split(r"[,\s]+", str(breeding or "")) if t in IOC_CODES]
        ranges[f"{genus} {epithet}".lower()] = ",".join(dict.fromkeys(codes))
    return ranges


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

    ranges = ioc_ranges(out.parent)
    rows, scientific, common, family, regions = [], [], [], [], []
    for i, (ranks, common_name) in enumerate(names):
        if len(ranks) == 7 and ranks[CLASS_RANK_IDX] == "Aves":
            rows.append(i)
            sci = f"{ranks[-2]} {ranks[-1]}"
            scientific.append(sci)
            common.append(title_case(common_name) if common_name else "")
            family.append(ranks[FAMILY_RANK_IDX])
            regions.append(ranges.get(sci.lower(), ""))
    named = sum(bool(c) for c in common)
    ranged = sum(bool(r) for r in regions)
    print(f"{len(rows)} Aves rows of {len(names)} | with common name: {named}")
    print(f"with IOC range: {ranged} ({ranged / len(rows):.0%}), the rest never filter out")

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
        regions=np.array(regions),
        logit_scale=np.float32(logit_scale),
    )
    print(f"saved {out} ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="models/species_labels.npz")
    args = parser.parse_args()
    build(Path(args.out))
