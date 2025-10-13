# MAISI VAE

Minimal PyTorch wrapper around NVIDIA’s MAISI autoencoder with built-in basic pre/post-processing and sliding-window encode/decode for large 3D volumes. This allows users to quickly generate latent CTs and MRs from full volumes and decode latent CT volumes into synthetic CTs with HU intensities. 

## What is MAISI?
MAISI (Medical AI for Synthetic Imaging) is NVIDIA’s 3D latent-diffusion framework for generating high-resolution synthetic CT (and a VAE that also supports MR). It provides flexible volumes/voxel spacing and optional mask-conditioned control. See the paper and official resources:
- Paper: [MAISI: Medical AI for Synthetic Imaging](https://arxiv.org/abs/2409.11169)
- Docs: [NVIDIA NIM for MAISI](https://docs.nvidia.com/nim/medical/maisi/latest/overview.html)
- Tutorials: [Project-MONAI/tutorials/generation/maisi](https://github.com/Project-MONAI/tutorials/tree/main/generation/maisi)

> [!WARNING]
> This repo only wraps the VAE part (encode/decode). It doesn’t run diffusion sampling.

## Features
- Encode/Decode with SlidingWindowInferer (configurable ROI, overlap, batch size).
- Simple CT/MR normalization, mask-aware preprocessing, and post-masking.
- Optional HU scaling for CT and SITK export with copied image metadata, and auto-download of the public MAISI VAE checkpoint.

## Install 

This process uses [uv](https://docs.astral.sh/uv/)

```bash
git clone git@github.com:sCT-Masters-Project/MaisiVAE.git
cd MaisiVAE
uv sync
source .venv/bin/activate
```

## Quickstart
```python 
import torch, SimpleITK as sitk
from MaisiVAE import MaisiVAE  # your file

# Read data (D,H,W)
ct = torch.from_numpy(sitk.GetArrayFromImage(sitk.ReadImage("ct.mha"))).float()
mr = torch.from_numpy(sitk.GetArrayFromImage(sitk.ReadImage("mr.mha"))).float()
mask = torch.from_numpy(sitk.GetArrayFromImage(sitk.ReadImage("mask.mha"))).float()

vae = MaisiVAE(config_dir="configs", progress_bar=True)

z_ct = vae.encode(ct, mask, modality="ct")
z_mr = vae.encode(mr, mask, modality="mr")

recon_ct, ct_img = vae.decode(z_ct, mask, ref_img_shape=ct.shape, ref_sitk=sitk.ReadImage("ct.mha"))
recon_mr, mr_img = vae.decode(z_mr, mask, ref_img_shape=mr.shape, ref_sitk=sitk.ReadImage("mr.mha"))
```

---
See MAISI paper and NVIDIA NIM docs linked above for details and licensing. 
