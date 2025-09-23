import os
import json
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from monai.apps import download_url
from monai.inferers.inferer import SlidingWindowInferer

import SimpleITK as sitk

# This import is expected to exist in your environment, as in your scripts.
# It should build the exact AutoEncoder architecture for the MAISI weights.
from scripts.utils import define_instance

class MaisiVAE:
    """
    Minimal MAISI VAE wrapper with encode/decode and built-in pre/post-processing.

    Key behaviors:
      - init optionally downloads only 'autoencoder_epoch273.pt' into weights_dir.
      - encode() expects a single volume (CT or MR) and a mask; returns latent (C,D,H,W).
      - decode() expects a latent and a mask; returns reconstructed tensor (D,H,W).
        If ref_sitk is provided, also returns a SimpleITK image with copied metadata.
      - No file I/O (reading/writing) inside this class.

    Shapes:
      - volume (input): torch.Tensor [D,H,W], float
      - mask (input):   torch.Tensor [D,H,W], float in {0,1} (will be internally resized)
      - latent:         torch.Tensor [C,D,H,W]
      - recon:          torch.Tensor [D,H,W]
    """

    def __init__(
        self,
        config_dir: str,
        weights_dir: str = "/local/scratch/dtaskiran/tmp/models",
        weight_filename: str = "autoencoder_epoch273.pt",
        download_weights: bool = True,
        gpu_id: int = 0,
        target_shape: Tuple[int, int, int] = (128, 512, 512),
        encode_roi: Tuple[int, int, int] = (32, 96, 32),
        decode_roi: Tuple[int, int, int] = (32, 96, 32),
        encode_overlap: float = 0.5,
        decode_overlap: float = 0.8,
        infer_mode: str = "gaussian",
        encode_sw_batch_size: int = 16,
        decode_sw_batch_size: int = 1,
        compile_model: bool = False,
        progress_bar: bool = False,
    ):
        """
        Args:
          config_dir: directory containing MAISI config files (expects 'config_maisi.json').
          weights_dir: where to place/load 'autoencoder_epoch273.pt'.
          weight_filename: filename (kept for flexibility).
          download_weights: if True, downloads weight if missing.
          gpu_id: CUDA device index (uses CPU if CUDA not available).
          target_shape: (D,H,W) for encoder pre-processing.
          encode_* / decode_*: Sliding-window settings.
          compile_model: torch.compile(autoencoder) if True (requrites PyTorch 2+).
        """
        self.config_dir = config_dir
        self.weights_dir = weights_dir
        self.weight_path = os.path.join(self.weights_dir, weight_filename)
        self.target_shape = target_shape
        self.progress_bar = progress_bar
        self.infer_mode = infer_mode

        os.makedirs(self.weights_dir, exist_ok=True)

        if download_weights and not os.path.exists(self.weight_path):
            download_url(
                url=(
                    "https://developer.download.nvidia.com/assets/Clara/monai/tutorials/"
                    "model_zoo/model_maisi_autoencoder_epoch273_alternative.pt"
                ),
                filepath=self.weight_path,
            )

        # Load model config for building the architecture
        with open(os.path.join(config_dir, "config_maisi.json"), "r") as f:
            model_def = json.load(f)

        # Merge into a simple args namespace required by define_instance
        class _A:  # lightweight, avoids argparse dependency here
            pass

        self.args = _A()
        for k, v in model_def.items():
            setattr(self.args, k, v)

        # Device
        if torch.cuda.is_available():
            self.device = torch.device(f"cuda:{gpu_id}")
        else:
            self.device = torch.device("cpu")

        # Build autoencoder and load weights
        self.autoencoder = define_instance(self.args, "autoencoder_def").to(self.device)
        ckpt = torch.load(self.weight_path, map_location=self.device, weights_only=True)
        self.autoencoder.load_state_dict(ckpt)
        self.autoencoder.eval()
        if compile_model:
            self.autoencoder = torch.compile(self.autoencoder)

        # Sliding window inferers
        self.enc_inferer = SlidingWindowInferer(
            roi_size=list(encode_roi),
            sw_batch_size=encode_sw_batch_size,
            overlap=encode_overlap,
            mode=self.infer_mode,
            sw_device=self.device,
            progress=self.progress_bar,
        )
        self.dec_inferer = SlidingWindowInferer(
            roi_size=list(decode_roi),
            sw_batch_size=decode_sw_batch_size,
            overlap=decode_overlap,
            mode=self.infer_mode,
            sw_device=self.device,
            progress=self.progress_bar,
        )

    @torch.inference_mode()
    def encode(
        self,
        volume: torch.Tensor,   # [D,H,W], float
        mask: torch.Tensor,     # [D,H,W], 0/1
        modality: str = "ct",   # "ct" or "mr"
        scale_factor: int = 1,   # optional upscaling before encoding (like your script)
    ) -> torch.Tensor:
        """
        Returns latent tensor [C,D,H,W].
        """
        x = self.preprocess(volume, mask, modality)  # -> [1,1,D,H,W] in [0,1]
        if scale_factor != 1:
            x = F.interpolate(x, scale_factor=scale_factor, mode="trilinear", align_corners=False)
        x = x.to(self.device)

        with torch.autocast(device_type=self.device.type if self.device.type == "cuda" else "cpu"):
            # autoencoder.encode returns (latent, *extras), mirror your code using tuple[0]
            latent_tuple = self.enc_inferer(network=self.autoencoder.encode, inputs=x)
            z = latent_tuple[0]  # [1,C,D,H,W] after SWI
        return z[0]  # [C,D,H,W]

    @torch.inference_mode()
    def decode(
        self,
        latent: torch.Tensor,       # [C,D,H,W], float
        mask: torch.Tensor,         # [D,H,W], 0/1
        ref_img_shape: Tuple[int, int, int],  # (D,H,W) target (e.g., MR shape)
        ref_sitk: Optional[sitk.Image] = None,
        erode_kernel: int = 3,
        hu_scale: Tuple[float, float] = (-1024.0, 1000.0),
    ):
        """
        Decodes a latent and postprocesses to original shape.
        Returns:
          recon_dhw: torch.Tensor [D,H,W], float (after masking + HU scaling)
          recon_sitk: Optional[SimpleITK.Image] if ref_sitk is provided (CopyInformation applied to match spacing, origin, direction)
        """
        # Prepare latent to 1×C×D×H×W
        z = latent.unsqueeze(0).to(self.device)

        with torch.autocast(device_type=self.device.type if self.device.type == "cuda" else "cpu"):
            out_tuple = self.dec_inferer(network=self.autoencoder.decode, inputs=z)
            recon = out_tuple[0]  # [1,1,D,H,W] or [1,D,H,W] depending on net; ensure [1,1,...]
            if recon.ndim == 4:
                recon = recon.unsqueeze(1)
            # Resize to ref shape and apply mask erosion & masking + HU scaling
            recon_dhw = self.postprocess(
                recon[0, 0],           # [D,H,W] (network output)
                mask,                  # original mask (any shape; will be resized)
                ref_img_shape,
                erode_kernel=erode_kernel,
                hu_scale=hu_scale,
            )

        if ref_sitk is None:
            return recon_dhw, None
        else:
            img = self.tensor_to_sitk(recon_dhw, ref_sitk)
            return recon_dhw, img

    def preprocess(
        self,
        volume: torch.Tensor,  # [D,H,W]
        mask: torch.Tensor,    # [D,H,W]
        modality: str,         # 'ct' or 'mr'
    ) -> torch.Tensor:
        """
        Resize to target_shape, set regions outside of the mask to 'air' in the native intensity domain,
        then normalize to [0,1], and return [1,1,D,H,W].

        Air values:
        - CT: -1000 HU
        - MR: 0 (assumed background air)

        For MR, normalization stats (min, 99.5% quantile) are computed on the IN-MASK voxels
        to avoid background bias.
        """
        assert volume.ndim == 3 and mask.ndim == 3, "volume/mask must be [D,H,W] tensors"

        tgt = self.target_shape
        vol = F.interpolate(volume[None, None, ...], size=tgt, mode="trilinear", align_corners=False)[0, 0]
        msk = F.interpolate(mask[None, None, ...].float(), size=tgt, mode="trilinear", align_corners=False)[0, 0]

        if modality.lower() == "ct":
            # Set outside mask to air in HU, then map [-1000,1000] -> [0,1]
            vol_air = -1000.0
            vol = torch.where(msk > 0, vol, torch.as_tensor(vol_air, dtype=vol.dtype, device=vol.device))
            x = self._normalize_to_range(vol, -1000.0, 1000.0)

        elif modality.lower() == "mr":
            # Set outside mask to air (0), compute stats on in-mask voxels, then normalize to [0,1]
            vol_air = 0.0
            vol = torch.where(msk > 0, vol, torch.as_tensor(vol_air, dtype=vol.dtype, device=vol.device))

            # Stats on in-mask to avoid background bias
            inmask = vol[msk > 0]
            if inmask.numel() == 0:
                # Degenerate mask; fall back to safe zero tensor
                x = torch.zeros_like(vol)
            else:
                vmin = torch.min(inmask)
                # sample for quantile if huge to save memory; here direct quantile on in-mask
                n = inmask.numel()
                k = int(0.995 * (n - 1)) + 1
                vmax = inmask.view(-1).kthvalue(k).values
                if torch.isclose(vmax, vmin):
                    x = torch.zeros_like(vol)
                else:
                    x = (vol - vmin) / (vmax - vmin)
                    x = torch.clamp(x, 0, 1)
        else:
            raise ValueError("modality must be 'ct' or 'mr'")

        return x[None, None, ...]  # [1,1,D,H,W]


    def postprocess(
        self,
        recon_dhw: torch.Tensor,    # [D,H,W] network output (assumed ~[0,1])
        mask: torch.Tensor,         # [D,H,W] original mask
        ref_img_shape: Tuple[int, int, int],
        erode_kernel: int = 5,
        hu_scale: Tuple[float, float] = (-1024.0, 1000.0),
    ) -> torch.Tensor:
        """
        Resize recon to ref_img_shape, apply eroded mask, and map to HU range.
        """
        # Resize recon to ref shape
        r = F.interpolate(recon_dhw[None, None, ...], size=ref_img_shape, mode="trilinear", align_corners=False)[0, 0]
        # Resize mask and erode
        m = F.interpolate(mask[None, None, ...].float().to(self.device), size=ref_img_shape, mode="trilinear", align_corners=False)[0, 0]
        m = self._erode_mask_3d(m, kernel_size=erode_kernel)

        r = r * m

        # Map [0,1] -> [hu_min, hu_max]
        hu_min, hu_max = hu_scale
        span = (hu_max - hu_min)
        r = r * span + hu_min
        return r  # [D,H,W]

    @staticmethod
    def tensor_to_sitk(tensor_dhw: torch.Tensor, ref_img: sitk.Image) -> sitk.Image:
        """
        Convert a [D,H,W] tensor to SimpleITK.Image and CopyInformation from ref_img.
        """
        arr = tensor_dhw.detach().cpu().numpy()
        img = sitk.GetImageFromArray(arr)
        img.CopyInformation(ref_img)
        return img

    @staticmethod
    def _normalize_to_range(x: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
        lo = float(lo); hi = float(hi)
        span = hi - lo
        if span == 0:
            return torch.zeros_like(x)
        out = (x - lo) / span
        return torch.clamp(out, 0, 1)

    @staticmethod
    def _erode_mask_3d(mask_dhw: torch.Tensor, kernel_size: int = 3) -> torch.Tensor:
        """
        Fast binary erosion using max-pooling trick.
        mask_dhw: [D,H,W] in {0,1}
        """
        m = mask_dhw[None, None, ...].float()          # [1,1,D,H,W]
        eroded = F.max_pool3d(-m, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
        eroded = -(eroded)
        m = (eroded > 0.5).float()
        return m[0, 0]

if __name__ == "__main__":
    ### Example usage with test data
    # Paths to your data
    ct_path = '/local/scratch/dtaskiran/tutorials/generation/maisi/1ABA011_ct.mha'
    mr_path = '/local/scratch/dtaskiran/tutorials/generation/maisi/1ABA011_mr.mha'
    mask_path = '/local/scratch/dtaskiran/tutorials/generation/maisi/1ABA011_mask.mha'

    # Read data using SimpleITK
    ct_sitk = sitk.ReadImage(ct_path)
    mr_sitk = sitk.ReadImage(mr_path)
    mask_sitk = sitk.ReadImage(mask_path)

    # Convert to torch tensors
    ct = torch.from_numpy(sitk.GetArrayFromImage(ct_sitk)).float()  # [D,H,W]
    mr = torch.from_numpy(sitk.GetArrayFromImage(mr_sitk)).float()  # [D,H,W]
    mask = torch.from_numpy(sitk.GetArrayFromImage(mask_sitk)).float()  # [D,H,W]
    print(f"CT shape: {ct.shape}, MR shape: {mr.shape}, Mask shape: {mask.shape}")

    # initialize MAISI VAE
    maisi_vae = MaisiVAE(config_dir='/local/scratch/dtaskiran/tutorials/generation/maisi/configs', 
                         gpu_id=2, progress_bar=True)

    # Encode CT and MR to latent
    latent_ct = maisi_vae.encode(torch.as_tensor(ct), torch.as_tensor(mask), modality='ct', scale_factor=1)
    latent_mr = maisi_vae.encode(torch.as_tensor(mr), torch.as_tensor(mask), modality='mr', scale_factor=1)
    print(f"CT latent shape: {latent_ct.shape}, MR latent shape: {latent_mr.shape}")

    # Save latent (isVector for multi-channel output)
    latent_ct_dhwc = latent_ct.permute(1, 2, 3, 0).cpu().numpy()  # [D,H,W,C]
    output_filename = '/local/scratch/dtaskiran/tutorials/generation/maisi/1ABA011_latent_ct.mha'
    image = sitk.GetImageFromArray(latent_ct_dhwc, isVector=True)
    sitk.WriteImage(image, output_filename)

    latent_mr_dhwc = latent_mr.permute(1, 2, 3, 0).cpu().numpy()  # [D,H,W,C]
    output_filename = '/local/scratch/dtaskiran/tutorials/generation/maisi/1ABA011_latent_mr.mha'
    image = sitk.GetImageFromArray(latent_mr_dhwc, isVector=True)
    sitk.WriteImage(image, output_filename)

    # Decode latent back to image space and save
    recon_ct_dhw, recon_ct_sitk = maisi_vae.decode(latent_ct, torch.as_tensor(mask), ct.shape, ref_sitk=ct_sitk)
    recon_mr_dhw, recon_mr_sitk = maisi_vae.decode(latent_mr, torch.as_tensor(mask), mr.shape, ref_sitk=mr_sitk)
    print(f"CT recon shape: {recon_ct_dhw.shape}, MR recon shape: {recon_mr_dhw.shape}")

    sitk.WriteImage(recon_ct_sitk, '/local/scratch/dtaskiran/tutorials/generation/maisi/1ABA011_recon_ct.mha')
    sitk.WriteImage(recon_mr_sitk, '/local/scratch/dtaskiran/tutorials/generation/maisi/1ABA011_recon_mr.mha')

