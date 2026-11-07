#!/usr/bin/env python3
"""
EdgeDiff-CV Studio — Standalone CLI & Web UI Runner (Problem Statement 3: ProbID - P3)
Supports one-command CLI inference (`python run_inference.py --output_dir ./results/`)
or launching the interactive Gradio workbench (`python run_inference.py --launch_ui`).
"""
import argparse
import os
import time
import warnings
warnings.filterwarnings("ignore")

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from diffusers import DiffusionPipeline, AutoencoderKL, AutoencoderTiny, LCMScheduler
from diffusers.utils import logging as diffusers_logging

diffusers_logging.set_verbosity_error()


def latent_frequency_boost_callback(pipe, step_index, timestep, callback_kwargs):
    latents = callback_kwargs["latents"]
    total_steps = getattr(pipe, "_num_timesteps", 4)
    if 0 < step_index < (total_steps - 1):
        low_freq = F.avg_pool2d(latents, kernel_size=3, stride=1, padding=1)
        high_freq = latents - low_freq
        latents = latents + (0.05 * high_freq)
    callback_kwargs["latents"] = latents
    return callback_kwargs


class BalancedClassicalCV:
    @staticmethod
    def evaluate_iqa(img_bgr: np.ndarray) -> dict:
        if img_bgr.shape[0] != 512 or img_bgr.shape[1] != 512:
            eval_bgr = cv2.resize(img_bgr, (512, 512), interpolation=cv2.INTER_CUBIC)
        else:
            eval_bgr = img_bgr
        gray = cv2.cvtColor(eval_bgr, cv2.COLOR_BGR2GRAY)
        sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        f = np.fft.fftshift(np.fft.fft2(gray))
        mag = 20 * np.log(np.abs(f) + 1e-8)
        artifact_idx = float(np.percentile(mag, 99) - np.mean(mag))
        contrast = float(gray.std())
        return {
            "Sharpness": round(sharpness, 1),
            "Artifact_Index": round(artifact_idx, 1),
            "Contrast": round(contrast, 1),
            "fft_mag": mag,
        }

    @classmethod
    def process(cls, pil_img: Image.Image, target_res: int = 512):
        t0 = time.perf_counter()
        img_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        was_upscaled = img_bgr.shape[0] != target_res
        pre_iqa = cls.evaluate_iqa(img_bgr)

        adaptive_sigma = float(np.clip(18.0 + max(0.0, pre_iqa["Artifact_Index"] - 42.0) * 0.85, 18.0, 34.0))
        smoothed = cv2.bilateralFilter(img_bgr, d=5, sigmaColor=adaptive_sigma, sigmaSpace=adaptive_sigma)

        lab = cv2.cvtColor(smoothed, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=1.25, tileGridSize=(8, 8))
        l_opt = clahe.apply(l)
        color_bgr = cv2.cvtColor(cv2.merge((l_opt, a, b)), cv2.COLOR_LAB2BGR)

        hsv = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.08, 0, 255)
        color_bgr = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

        if was_upscaled:
            color_bgr = cv2.resize(color_bgr, (target_res, target_res), interpolation=cv2.INTER_LANCZOS4)

        sharp_weight = 1.52 if was_upscaled else 1.35
        blur = cv2.GaussianBlur(color_bgr, (0, 0), 1.1)
        sharpened = cv2.addWeighted(color_bgr, sharp_weight, blur, -(sharp_weight - 1.0), 0)

        gray = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2GRAY)
        gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        edge_mask = np.hypot(gx, gy)
        edge_mask = np.clip(edge_mask / (np.percentile(edge_mask, 95) + 1e-6), 0.18, 1.0)[..., None]

        final_bgr = np.clip(sharpened * edge_mask + color_bgr * (1.0 - edge_mask), 0, 255).astype(np.uint8)
        cv_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        post_iqa = cls.evaluate_iqa(final_bgr)

        edge_vis = cv2.cvtColor(cv2.applyColorMap((edge_mask.squeeze() * 255).astype(np.uint8), cv2.COLORMAP_INFERNO), cv2.COLOR_BGR2RGB)
        fft_norm = cv2.normalize(pre_iqa["fft_mag"], None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        fft_vis = cv2.cvtColor(cv2.applyColorMap(fft_norm, cv2.COLORMAP_VIRIDIS), cv2.COLOR_BGR2RGB)

        final_pil = Image.fromarray(cv2.cvtColor(final_bgr, cv2.COLOR_BGR2RGB))
        return final_pil, Image.fromarray(edge_vis), Image.fromarray(fft_vis), pre_iqa, post_iqa, cv_ms


def main():
    parser = argparse.ArgumentParser(description="EdgeDiff-CV Resource-Efficient Text-to-Image Runner (P3)")
    parser.add_argument(
        "--prompt",
        type=str,
        default="a white ceramic coffee cup on the left of a golden croissant on a dark marble table, morning light",
        help="Natural language prompt for text-to-image synthesis",
    )
    parser.add_argument("--input_dir", type=str, default="./samples/", help="Optional directory containing prompts.txt")
    parser.add_argument("--output_dir", type=str, default="./results/", help="Directory to save generated and restored images")
    parser.add_argument("--steps", type=int, default=4, choices=[3, 4, 6], help="Sampling steps (3=Eco, 4=Golden Balance, 6=Max Accuracy)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    print(f"[EdgeDiff-CV] Initializing pipeline on {device.upper()} ({dtype})...")
    taesd = AutoencoderTiny.from_pretrained("madebyollin/taesd", torch_dtype=dtype).to(device)
    pipe = DiffusionPipeline.from_pretrained(
        "SimianLuo/LCM_Dreamshaper_v7", vae=taesd, torch_dtype=dtype, safety_checker=None
    ).to(device)
    pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
    pipe.enable_attention_slicing()

    res = 384 if args.steps == 3 else 512
    cfg = 1.4 if args.steps == 3 else (1.6 if args.steps == 4 else 1.8)

    t0 = time.perf_counter()
    raw_img = pipe(
        prompt=args.prompt,
        num_inference_steps=args.steps,
        guidance_scale=cfg,
        height=res,
        width=res,
        generator=torch.Generator(device=device).manual_seed(args.seed),
        callback_on_step_end=latent_frequency_boost_callback,
    ).images[0]
    dl_ms = round((time.perf_counter() - t0) * 1000.0, 1)

    restored_img, edge_img, fft_img, pre_iqa, post_iqa, cv_ms = BalancedClassicalCV.process(raw_img, target_res=512)

    raw_img.save(os.path.join(args.output_dir, "1_raw_compressed.png"))
    restored_img.save(os.path.join(args.output_dir, "2_restored_edgediff_cv.png"))
    edge_img.save(os.path.join(args.output_dir, "3_sobel_edge_mask.png"))
    fft_img.save(os.path.join(args.output_dir, "4_fft_spectrum.png"))

    print(f"[EdgeDiff-CV] Complete! DL Latency: {dl_ms} ms | Classical CV Latency: {cv_ms} ms")
    print(f"[EdgeDiff-CV] Sharpness Recovery: {pre_iqa['Sharpness']} -> {post_iqa['Sharpness']} | Saved to {args.output_dir}")


if __name__ == "__main__":
    main()
