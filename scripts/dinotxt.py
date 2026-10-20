"""Loads dino.txt (Meta's DINOv3 ViT-L/16 LVD-1689M backbone + dino.txt vision head and text encoder) in PyTorch.

Uses existing weights/ files or downloads pinned weights into the Hugging Face
cache. Authenticate with `hf auth login` or HF_TOKEN for Meta's gated backbone.
"""

import sys
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "dinov3"))

from dinov3.hub.dinotxt import dinov3_vitl16_dinotxt_tet1280d20h24l  # noqa: E402

WEIGHTS = ROOT / "weights"
HF_BACKBONE = WEIGHTS / "hf-dinov3-vitl16-lvd1689m" / "model.safetensors"
DINOTXT_HEAD = WEIGHTS / "dinov3_vitl16_dinotxt_vision_head_and_text_encoder-a442d8f5.pth"
BPE_VOCAB = WEIGHTS / "bpe_simple_vocab_16e6.txt.gz"
BACKBONE_REPO = "facebook/dinov3-vitl16-pretrain-lvd1689m"
BACKBONE_REVISION = "ea8dc2863c51be0a264bab82070e3e8836b02d51"
HEAD_REPO = "PIA-SPACE-LAB/dinov3_vitl16_dinotxt_vision_head_and_text_encoder"
HEAD_REVISION = "182463bf05a7b7ecb3bd9b4bf2a026ccb4054931"
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def model_paths() -> tuple[Path, Path, Path]:
    """Resolve backbone, head, and tokenizer without downloading on import."""
    files = (
        (HF_BACKBONE, BACKBONE_REPO, BACKBONE_REVISION),
        (DINOTXT_HEAD, HEAD_REPO, HEAD_REVISION),
        (BPE_VOCAB, HEAD_REPO, HEAD_REVISION),
    )
    return tuple(
        path if path.is_file() else Path(hf_hub_download(
            repo_id=repo, filename=path.name, revision=revision,
        ))
        for path, repo, revision in files
    )


def hf_backbone_to_dinov3(hf: dict[str, torch.Tensor], ref: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Map Hugging Face DINOv3ViTModel weights onto the original dinov3 repo layout.

    `ref` is the target module's state dict; its buffers (rope periods, k-bias mask) are kept as initialised.
    """
    sd = {
        "cls_token": hf["embeddings.cls_token"],
        "storage_tokens": hf["embeddings.register_tokens"],
        "mask_token": hf["embeddings.mask_token"].reshape(ref["mask_token"].shape),
        "patch_embed.proj.weight": hf["embeddings.patch_embeddings.weight"],
        "patch_embed.proj.bias": hf["embeddings.patch_embeddings.bias"],
        "norm.weight": hf["norm.weight"],
        "norm.bias": hf["norm.bias"],
        "rope_embed.periods": ref["rope_embed.periods"],
    }
    n_layers = 1 + max(int(k.split(".")[1]) for k in hf if k.startswith("layer."))
    for i in range(n_layers):
        h, o = f"layer.{i}.", f"blocks.{i}."
        q, k, v = (hf[h + f"attention.{p}_proj.weight"] for p in "qkv")
        qb, vb = hf[h + "attention.q_proj.bias"], hf[h + "attention.v_proj.bias"]
        sd |= {
            o + "norm1.weight": hf[h + "norm1.weight"],
            o + "norm1.bias": hf[h + "norm1.bias"],
            o + "attn.qkv.weight": torch.cat([q, k, v]),
            o + "attn.qkv.bias": torch.cat([qb, torch.zeros_like(qb), vb]),
            o + "attn.qkv.bias_mask": ref[o + "attn.qkv.bias_mask"],
            o + "attn.proj.weight": hf[h + "attention.o_proj.weight"],
            o + "attn.proj.bias": hf[h + "attention.o_proj.bias"],
            o + "ls1.gamma": hf[h + "layer_scale1.lambda1"],
            o + "norm2.weight": hf[h + "norm2.weight"],
            o + "norm2.bias": hf[h + "norm2.bias"],
            o + "mlp.fc1.weight": hf[h + "mlp.up_proj.weight"],
            o + "mlp.fc1.bias": hf[h + "mlp.up_proj.bias"],
            o + "mlp.fc2.weight": hf[h + "mlp.down_proj.weight"],
            o + "mlp.fc2.bias": hf[h + "mlp.down_proj.bias"],
            o + "ls2.gamma": hf[h + "layer_scale2.lambda1"],
        }
    return sd


def load_dinotxt():
    """Return (model, tokenizer) for dino.txt on the official DINOv3 ViT-L/16 LVD-1689M backbone."""
    backbone_path, head_path, vocab_path = model_paths()
    model, tokenizer = dinov3_vitl16_dinotxt_tet1280d20h24l(pretrained=False, bpe_path_or_url=str(vocab_path))
    backbone = model.visual_model.backbone
    backbone.load_state_dict(hf_backbone_to_dinov3(load_file(backbone_path), backbone.state_dict()), strict=True)

    head = torch.load(head_path, map_location="cpu", weights_only=True)
    missing, unexpected = model.load_state_dict(head, strict=False)
    assert not unexpected, unexpected
    assert all(k.startswith("visual_model.backbone.") for k in missing), [
        k for k in missing if not k.startswith("visual_model.backbone.")
    ]
    return model.eval(), tokenizer
