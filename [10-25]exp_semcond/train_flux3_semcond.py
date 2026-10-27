#!/usr/bin/env python
# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
#
# Seg2Any の学習スクリプト (train.py) を FLUX 3 Image Klein に移植したもの (train_flux2.py の FLUX.3 版)。
# 変更点は FLUX3_PORT.md の Step 4 を参照。timestep のサンプリング (shift 6.93 の固定 shift、weighting none) は
# diffusers 公式の examples/dreambooth/train_dreambooth_lora_flux3_klein.py と同じ。

import os
import sys
import argparse
# [10-25]exp_semcond：意味付き cond トークン版。train_flux3.py をコピーし、"semcond" と書いた箇所だけ変えた
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import copy
import logging
import math
import os
import json
import random
import shutil
from contextlib import nullcontext
from pathlib import Path

import time
import accelerate
import numpy as np
import cv2
import torch
import torch.utils.checkpoint
from torch.nn import functional as F
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedType, ProjectConfiguration, DataLoaderConfiguration, set_seed, DistributedDataParallelKwargs, InitProcessGroupKwargs
from datetime import timedelta
from packaging import version
from peft import LoraConfig, set_peft_model_state_dict
from peft.utils import get_peft_model_state_dict
from PIL import Image
from tqdm.auto import tqdm
from safetensors.torch import load_file

from omegaconf import OmegaConf,DictConfig

import diffusers
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.optimization import get_scheduler
from diffusers.training_utils import cast_training_params,compute_density_for_timestep_sampling, compute_loss_weighting_for_sd3, free_memory
from diffusers.utils import check_min_version, is_wandb_available
from diffusers.utils.torch_utils import is_compiled_module

from src.models.transformer_flux3 import Flux3RegionalTransformer2DModel
from src.pipelines.pipeline_flux3 import SCHEDULER_SHIFT, TEXT_PAD_MULTIPLE, Flux3RegionalPipeline, prepare_image_ids
from dataset.collate_fn import collate_fn
from semcond import enable_semcond, save_sem_proj, load_sem_proj, region_pooled_embeds, cond_semantic  # semcond
from dataset.group_sampler import GroupSampler
from utils.utils import instantiate_from_config
from utils.visualizer import Visualizer

if is_wandb_available():
    import wandb

check_min_version("0.40.0.dev0")

logger = get_logger(__name__)

visualizer = Visualizer()

def encode_images(pixels: torch.Tensor, vae: torch.nn.Module, weight_dtype):
    """[-1,1] の画像 -> (B, 128, H/16, W/16)。推論側 (Flux3RegionalPipeline.encode_vae_image) と同じ処理:
    latent_dist.mode() -> 2x2 パッチ化 -> vae.bn の統計で正規化"""
    pixel_latents = vae.encode(pixels.to(vae.dtype)).latent_dist.mode()
    pixel_latents = Flux3RegionalPipeline._patchify_latents(pixel_latents)
    bn_mean = vae.bn.running_mean.view(1, -1, 1, 1).to(pixel_latents.device, pixel_latents.dtype)
    bn_std = torch.sqrt(vae.bn.running_var.view(1, -1, 1, 1) + vae.config.batch_norm_eps).to(
        pixel_latents.device, pixel_latents.dtype
    )
    pixel_latents = (pixel_latents - bn_mean) / bn_std
    return pixel_latents.to(weight_dtype)

def unpack_latents(x: torch.Tensor, height: int, width: int):
    """Flux3RegionalPipeline.pack_latents の逆: (B, H*W, C) -> (B, C, H, W)"""
    return x.permute(0, 2, 1).reshape(x.shape[0], x.shape[2], height, width)

# FLUX.3 の変調 Linear はブロックの外に、モードごと (txt / image / image_cond) × 前半用・後半用にある。
# default (txt + noisy img 用) には txt と image の分だけ入れる。image_cond の分は cond_lora_layers 側で指定する
MODULATION_LINEARS = [
    "early_stream_modulations.txt.linear",
    "early_stream_modulations.image.linear",
    "single_stream_modulations.txt.linear",
    "single_stream_modulations.image.linear",
]

# default LoRA を付ける「DiT ブロック」: txt と noisy img の mode blocks と joint blocks。
# image_cond の mode blocks は cond 専用の重みなので含めない (cond_lora_layers 側)
DIT_BLOCK_PREFIXES = ("txt_mode_blocks.", "mode_blocks.image.", "single_transformer_blocks.")

def get_lora_target_modules(lora_layers, flux_transformer):
    if lora_layers is not None:
        if lora_layers == "all-linear":
            target_modules = set()
            for name, module in flux_transformer.named_modules():
                if isinstance(module, torch.nn.Linear):
                    target_modules.add(name)
            target_modules = list(target_modules)
        elif lora_layers in ("all-linear-in-dit-blocks", "all-linear-in-dit-blocks+modulation"):
            target_modules = set()
            for name, module in flux_transformer.named_modules():
                if name.startswith(DIT_BLOCK_PREFIXES):
                    if isinstance(module, torch.nn.Linear):
                        target_modules.add(name)
            if lora_layers.endswith("+modulation"):
                # FLUX.1 の all-linear-in-dit-blocks はブロック内の norm*.linear (変調) も含んでいた。それに揃える
                target_modules.update(MODULATION_LINEARS)
            target_modules = list(target_modules)
        elif lora_layers.startswith("regular_expression:"):
            target_modules = lora_layers[len("regular_expression:"):]
        else:
            target_modules = [layer.strip() for layer in lora_layers.split(",")]
    else:
        target_modules = [
            "attn.to_k",
            "attn.to_q",
            "attn.to_v",
            "attn.to_out.0",
            "ff.linear_in",
            "ff.linear_out",
        ]
    return target_modules

def log_validation(flux_transformer, text_encoding_pipeline, args, val_dataloader, accelerator, weight_dtype, step):
    logger.info("Running validation... ")

    flux_transformer = accelerator.unwrap_model(flux_transformer)

    # Reuse the text encoder and VAE that are already loaded rather than pulling a second
    # copy of Qwen3-VL (~8.9GB) off disk onto the validating GPU.
    pipeline = Flux3RegionalPipeline(
        transformer=flux_transformer,
        vae=text_encoding_pipeline.vae,
        text_encoder=text_encoding_pipeline.text_encoder,
        tokenizer=text_encoding_pipeline.tokenizer,
        scheduler=FlowMatchEulerDiscreteScheduler(shift=SCHEDULER_SHIFT),
    )

    pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)

    if args.seed is None:
        generator = None
    else:
        generator = torch.Generator(device=accelerator.device).manual_seed(args.seed)

    image_logs = []
 
    autocast_ctx = torch.autocast(accelerator.device.type, weight_dtype)
    num_validation_images = min(args.trainer.num_validation_images,len(val_dataloader))
    for i,batch in enumerate(val_dataloader):
        # note: val_dataloader bs == 1
        images = []

        with autocast_ctx:
            image = pipeline(
                global_prompt = batch["global_caption"],
                regional_prompts = batch["regional_captions"],
                regional_labels = batch["label"],
                cond = (batch["cond_pixel_values"]+1)/2.0 if args.model.is_use_cond_token else None, # denormalize
                attention_mask_method = args.model.attention_mask_method,
                is_filter_cond_token=args.model.is_filter_cond_token,
                hard_attn_block_range = args.model.hard_attn_block_range,
                height = batch["pixel_values"].shape[-2],
                width = batch["pixel_values"].shape[-1],
                cond_scale_factor = args.cond_scale_factor,
                num_images_per_prompt = 1,
                guidance_scale = args.eval.guidance_scale,
                num_inference_steps = args.model.num_inference_steps,
                generator = generator,
                max_sequence_length = args.model.max_sequence_length,
                regional_max_sequence_length = args.model.regional_max_sequence_length,
                regional_text_position = args.model.regional_text_position,
                text_pad_multiple = args.model.get("text_pad_multiple", TEXT_PAD_MULTIPLE),
            ).images[0]
        images.append(image)
        
        gt_image = Image.open(batch['image_path'][0]).convert('RGB')
        gt_image = gt_image.resize(image.size, resample=Image.BICUBIC)
        gt_image = np.array(gt_image)
        
        image_with_label = np.array(image)
        image_with_label = cv2.cvtColor(image_with_label,cv2.COLOR_RGB2BGR)

        label = batch["label"][0]
        label = F.interpolate(label[None].float(), size=image_with_label.shape[:2], mode='nearest-exact')
        label = label[0, ...].long()
        label= label.cpu().numpy()
        image_with_label = visualizer.draw_binary_mask_with_caption(image_with_label, label, batch["regional_captions"][0], alpha=0.4)
        image_with_label = cv2.cvtColor(image_with_label,cv2.COLOR_BGR2RGB)
        
        image_logs.append(
            {"ground_truth": gt_image,"image_with_label":image_with_label, "images": images, "global_caption": batch["global_caption"][0] if batch["global_caption"] is not None else str(i)}
        )
        
        if i == num_validation_images:
            break

    tracker_key = "validation"
    for tracker in accelerator.trackers:
        if tracker.name == "tensorboard":
            for i,log in enumerate(image_logs):
                images = log["images"]
                global_caption = log["global_caption"]
                ground_truth = log["ground_truth"]
                image_with_label = log["image_with_label"]
                formatted_images = [ground_truth,image_with_label]
                for image in images:
                    formatted_images.append(np.asarray(image))
                formatted_images = np.stack(formatted_images)
                tracker.writer.add_images(global_caption, formatted_images, step, dataformats="NHWC")

        elif tracker.name == "wandb":
            formatted_images = []
            for i,log in enumerate(image_logs):
                images = log["images"]
                global_caption = log["global_caption"]
                ground_truth = log["ground_truth"]
                image_with_label = log["image_with_label"]
                formatted_images.append(wandb.Image(ground_truth, caption="ground_truth image"))
                formatted_images.append(wandb.Image(image_with_label, caption="image_with_label image"))
                for image in images:
                    image = wandb.Image(image, caption=global_caption)
                    formatted_images.append(image)

            tracker.log({tracker_key: formatted_images})
        else:
            logger.warning(f"image logging not implemented for {tracker.name}")
        
    del pipeline
    free_memory()
    return image_logs

def get_parser(**parser_kwargs):
    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument(
        "base",
        nargs="*",
        metavar="base_config.yaml",
        help="paths to base configs. Loaded from left-to-right. "
             "Parameters can be overwritten or added with command-line options of the form `--key value`.",
        default=list(),
    )
    return parser

def main(args):
    
    if torch.backends.mps.is_available() and args.mixed_precision == "bf16":
        # due to pytorch#99272, MPS does not yet support bfloat16.
        raise ValueError(
            "Mixed precision training with bfloat16 is not supported on MPS. Please use fp16 (recommended) or fp32 instead."
        )
    logging_dir = Path(args.project.output_dir, args.project.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.project.output_dir, logging_dir=logging_dir)
    init_process_group_kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=3600)) #  nccl timeout
    accelerator = Accelerator(
        gradient_accumulation_steps=args.trainer.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.project.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[init_process_group_kwargs],
    )

    # Disable AMP for MPS. A technique for accelerating machine learning computations on iOS and macOS devices.
    if torch.backends.mps.is_available():
        logger.info("MPS is enabled. Disabling AMP.")
        accelerator.native_amp = False

    if args.project.report_to == "wandb":
        if not is_wandb_available():
            raise ImportError("Make sure to install wandb if you want to use it for logging during training.")

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.project.output_dir is not None:
            os.makedirs(args.project.output_dir, exist_ok=True)
            
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Load models.
    # Measured on gp38 (251GB RAM, 8 ranks): loading the transformer without torch_dtype
    # upcasts the bf16 checkpoint to fp32 and pins 44.9GB of host RAM per rank (48.4GB
    # peak) -- ~388GB across 8 ranks, which OOMs the whole node. Passing torch_dtype lets
    # safetensors mmap the weights instead: 10.4GB peak per rank, falling back to 1.1GB
    # resident once they are on the GPU. Ranks are still let through load_group_size at a
    # time so the transient peak stays small.
    vae = None
    text_encoding_pipeline = None
    flux_transformer = None
    load_group_size = max(1, int(args.trainer.get("load_group_size", 2)))
    for load_group in range(math.ceil(accelerator.num_processes / load_group_size)):
        if accelerator.process_index // load_group_size == load_group:
            logger.info(f"rank {accelerator.process_index}: loading models")
            # Create a pipeline for text encoding (Qwen3-VL). We will move this pipeline to GPU/CPU as needed.
            # FLUX.3 の repo は部品を個別に読むので、VAE もこのパイプラインが持つものを fp32 にして使う
            text_encoding_pipeline = Flux3RegionalPipeline.from_pretrained(
                args.model.pretrained_model_name_or_path, transformer=None, torch_dtype=weight_dtype
            )
            text_encoding_pipeline.text_encoder.requires_grad_(False)
            text_encoding_pipeline.text_encoder.eval()
            vae = text_encoding_pipeline.vae

            # torch_dtype matters: the checkpoint is bf16 on disk, so omitting it upcasts to
            # fp32 only to cast straight back to bf16 below.
            flux_transformer = Flux3RegionalTransformer2DModel.from_pretrained(
                args.model.pretrained_model_name_or_path, subfolder="transformer", torch_dtype=weight_dtype
            )

            # Move vae, transformer and text_encoding_pipeline to device.
            # transformer は dtype を指定せずに移すだけにする: from_pretrained が QK norm (_keep_in_fp32_modules) を
            # fp32 のまま残しているので、.to(dtype=bf16) するとそれが bf16 に落ちて推論側とずれる
            text_encoding_pipeline = text_encoding_pipeline.to(accelerator.device)
            vae.to(accelerator.device, dtype=torch.float32)
            flux_transformer.to(accelerator.device)
        accelerator.wait_for_everyone()

    vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
    if args.model.hard_attn_block_range is None:
        args.model.hard_attn_block_range = flux_transformer.default_hard_attn_block_range()

    # FLUX.3 の repo には scheduler が無い。公式 (ModularPipeline / 公式の LoRA 学習スクリプト) と同じく shift 6.93 固定で作る
    noise_scheduler = FlowMatchEulerDiscreteScheduler(shift=SCHEDULER_SHIFT, use_dynamic_shifting=False)
    noise_scheduler_copy = copy.deepcopy(noise_scheduler)

    logger.info("All models loaded successfully")

    vae.requires_grad_(False)
    flux_transformer.requires_grad_(False)
    flux_transformer.eval()


    default_target_modules = get_lora_target_modules(args.model.default_lora_layers, flux_transformer)
    cond_target_modules = get_lora_target_modules(args.model.cond_lora_layers, flux_transformer)
    
    default_lora_config = LoraConfig(
        r=args.model.rank,
        lora_alpha=args.model.rank,
        init_lora_weights="gaussian" if args.model.gaussian_init_lora else True,
        target_modules=default_target_modules,
        lora_bias=args.model.use_lora_bias,
    )
    flux_transformer.add_adapter(default_lora_config, adapter_name='default') 

    if args.model.is_use_cond_token: 
        cond_lora_config = LoraConfig(
            r=args.model.rank,
            lora_alpha=args.model.rank,
            init_lora_weights="gaussian" if args.model.gaussian_init_lora else True,
            target_modules=cond_target_modules,
            lora_bias=args.model.use_lora_bias,
        )
        flux_transformer.add_adapter(cond_lora_config, adapter_name='cond')
        flux_transformer.set_adapter(['cond','default'])
        
    # semcond: ゼロ初期化の cond_sem_proj を付ける (LoRA を付けたあとに。学習対象になる)
    use_semcond = bool(args.model.get("semcond", False))
    if use_semcond:
        assert args.model.is_use_cond_token and args.model.is_filter_cond_token
        enable_semcond(flux_transformer)

    if args.trainer.gradient_checkpointing:
        flux_transformer.enable_gradient_checkpointing()
        
    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    # `accelerate` 0.16.0 will have better support for customized saving
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):

        def save_model_hook(models, weights, output_dir):
            if accelerator.is_main_process:
                cond_lora_layers_to_save = None
                for model in models:
                    if isinstance(unwrap_model(model), type(unwrap_model(flux_transformer))):
                        transformer_ = unwrap_model(model)
                        
                        default_lora_layers_to_save = get_peft_model_state_dict(transformer_, adapter_name="default")
                        
                        if "cond" in transformer_.peft_config: 
                            cond_lora_layers_to_save = get_peft_model_state_dict(transformer_, adapter_name="cond")
                        
                    else:
                        raise ValueError(f"unexpected save model: {model.__class__}")

                    # make sure to pop weight so that corresponding model is not saved again
                    if weights:
                        weights.pop()
                
                Flux3RegionalPipeline.save_lora_weights(
                    os.path.join(output_dir,'default'),
                    transformer_lora_layers=default_lora_layers_to_save,
                )
                if cond_lora_layers_to_save is not None:
                    Flux3RegionalPipeline.save_lora_weights(
                        os.path.join(output_dir,'cond'),
                        transformer_lora_layers=cond_lora_layers_to_save,
                    )
                if use_semcond:  # semcond
                    save_sem_proj(transformer_, output_dir)

        def load_model_hook(models, input_dir):
            transformer_ = None                
            if not accelerator.distributed_type == DistributedType.DEEPSPEED:
                while len(models) > 0:
                    model = models.pop()

                    if isinstance(model, type(unwrap_model(flux_transformer))):
                        transformer_ = model
                    else:
                        raise ValueError(f"unexpected save model: {model.__class__}")
            else:
                # when use DEEPSPEED, the model is not loaded here. Accelerator will load model automatically.
                # Here, we only validates that the checkpoint weights are correctly formatted. 
                transformer_ = Flux3RegionalTransformer2DModel.from_pretrained(
                    args.model.pretrained_model_name_or_path, subfolder="transformer"
                ).to(accelerator.device, weight_dtype)
                transformer_.add_adapter(default_lora_config, adapter_name='default')
                if args.model.is_use_cond_token: 
                    transformer_.add_adapter(cond_lora_config, adapter_name='cond')
                    transformer_.set_adapter(['cond','default'])
                            
            # load transformer
            lora_state_dict = Flux3RegionalPipeline.lora_state_dict(os.path.join(input_dir,'default'))
            transformer_lora_state_dict = {
                f'{k.replace("transformer.", "")}': v
                for k, v in lora_state_dict.items()
                if k.startswith("transformer.") and "lora" in k
            }
            incompatible_keys = set_peft_model_state_dict(
                transformer_, transformer_lora_state_dict, adapter_name="default"
            )
            if incompatible_keys is not None:
                # check only for unexpected keys
                unexpected_keys = getattr(incompatible_keys, "unexpected_keys", None)
                if unexpected_keys:
                    logger.warning(
                        f"Loading adapter weights from state_dict led to unexpected keys not found in the model: "
                        f" {unexpected_keys}. "
                    )
            
            if os.path.exists(os.path.join(input_dir,'cond')):
                lora_state_dict = Flux3RegionalPipeline.lora_state_dict(os.path.join(input_dir,'cond'))
                transformer_lora_state_dict = {
                    f'{k.replace("transformer.", "")}': v
                    for k, v in lora_state_dict.items()
                    if k.startswith("transformer.") and "lora" in k
                }
                incompatible_keys = set_peft_model_state_dict(
                    transformer_, transformer_lora_state_dict, adapter_name="cond"
                )
                if incompatible_keys is not None:
                    # check only for unexpected keys
                    unexpected_keys = getattr(incompatible_keys, "unexpected_keys", None)
                    if unexpected_keys:
                        logger.warning(
                            f"Loading adapter weights from state_dict led to unexpected keys not found in the model: "
                            f" {unexpected_keys}. "
                        )

            if use_semcond:  # semcond
                load_sem_proj(transformer_, input_dir)

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.trainer.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.trainer.scale_lr:
        args.trainer.learning_rate = (
            args.trainer.learning_rate * args.trainer.gradient_accumulation_steps * args.trainer.train_batch_size * accelerator.num_processes
        )

    optimizer_class = torch.optim.AdamW

    # Optimization parameters
    params_group = [{
        'params':list(filter(lambda p: p.requires_grad, flux_transformer.parameters())),
        'lr':args.trainer.learning_rate
    }]

    optimizer = optimizer_class(
        params_group,
        lr=args.trainer.learning_rate,
        betas=(args.optimizer.adam_beta1, args.optimizer.adam_beta2),
        weight_decay=args.optimizer.adam_weight_decay,
        eps=args.optimizer.adam_epsilon,
    )
    
    train_dataset = instantiate_from_config(args.data.train)
    val_dataset = instantiate_from_config(args.data.val)
    
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        sampler=GroupSampler(train_dataset,samples_per_gpu=args.trainer.train_batch_size * accelerator.num_processes),
        collate_fn=collate_fn,
        batch_size=args.trainer.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )
    
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        shuffle=False,
        collate_fn=collate_fn,
        batch_size=1,
        num_workers=args.dataloader_num_workers,
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.trainer.gradient_accumulation_steps)
    if args.trainer.max_train_steps is None:
        args.trainer.max_train_steps = args.trainer.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.scheduler.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.scheduler.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.trainer.max_train_steps * accelerator.num_processes,
        num_cycles=args.scheduler.lr_num_cycles,
        power=args.scheduler.lr_power,
    )
    # Prepare everything with our `accelerator`.
    flux_transformer, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        flux_transformer, optimizer, train_dataloader, lr_scheduler
    )
    unwrap_flux_transformer = unwrap_model(flux_transformer)
    
    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.trainer.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.trainer.max_train_steps = args.trainer.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.trainer.num_train_epochs = math.ceil(args.trainer.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_config = json.dumps(OmegaConf.to_container(args, resolve=True))
        tracker_config = {"tracker_config":tracker_config}
        accelerator.init_trackers(args.project.tracker_project_name, config=tracker_config)

    # Train!
    total_batch_size = args.trainer.train_batch_size * accelerator.num_processes * args.trainer.gradient_accumulation_steps
    
    if accelerator.is_main_process:
        trainable_params = [p for p in unwrap_flux_transformer.parameters() if p.requires_grad]
        total_params_count = sum(p.numel() for p in unwrap_flux_transformer.parameters())
        trainable_params_count = sum(p.numel() for p in trainable_params)
        
        print("\n====== flux transformers Parameter Statistics ======")
        print(f"Total Parameters: {total_params_count}, ({total_params_count/1e6:.2f}M)")
        print(f"Trainable Parameters: {trainable_params_count}, ({trainable_params_count/1e6:.2f}M)")
        print(f"Trainable %: {trainable_params_count/total_params_count*100:.4f}%")
        print("==================================\n")

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.trainer.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.trainer.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.trainer.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.trainer.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.project.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            logger.info(f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run.")
            args.resume_from_checkpoint = None
        else:
            logger.info(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.project.output_dir, path))
            global_step = int(path.split("-")[1])
            first_epoch = global_step // num_update_steps_per_epoch
            resume_global_step = global_step * args.trainer.gradient_accumulation_steps
            resume_step = resume_global_step % (num_update_steps_per_epoch * args.trainer.gradient_accumulation_steps)
    
    progress_bar = tqdm(
        range(0, args.trainer.max_train_steps),
        initial=global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    def get_sigmas(timesteps, n_dim=4, dtype=torch.float32):
        sigmas = noise_scheduler_copy.sigmas.to(device=accelerator.device, dtype=dtype)
        schedule_timesteps = noise_scheduler_copy.timesteps.to(accelerator.device)
        timesteps = timesteps.to(accelerator.device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma
    
    image_logs = None
    for epoch in range(first_epoch, args.trainer.num_train_epochs):
        
        train_dataloader.set_epoch(epoch)
        
        if args.resume_from_checkpoint and epoch == first_epoch:
            # Skip steps until we reach the resumed step
            work_dataloader = accelerate.skip_first_batches(train_dataloader,num_batches=resume_step)
        else:
            work_dataloader = train_dataloader
            
        for step, batch in enumerate(work_dataloader):
            with accelerator.accumulate(flux_transformer):                
                # vae encode
                pixel_latents = encode_images(batch["pixel_values"], vae.to(accelerator.device), weight_dtype)
                bsz = pixel_latents.shape[0]           
                
                cond_pixel_latents = None
                cond_ids= None
                if args.model.is_use_cond_token:
                    cond_latents = encode_images(batch["cond_pixel_values"], vae.to(accelerator.device), weight_dtype)
                    cond_pixel_latents = Flux3RegionalPipeline.pack_latents(cond_latents)  # B, h*w, 128
                    assert batch["pixel_values"].shape[-2] // batch["cond_pixel_values"].shape[-2] == batch["pixel_values"].shape[-1] // batch["cond_pixel_values"].shape[-1]
                    assert batch["pixel_values"].shape[-1] % batch["cond_pixel_values"].shape[-1] == 0
                    # (h*w, 4) = (0, 0.5h, 0.5w, -1)。参照画像 1 枚目と同じ l=-1。
                    # noisy latent と同じ空間座標に揃えるため cond_scale_factor 倍する
                    cond_ids = prepare_image_ids(
                        *cond_latents.shape[-2:], l_coord=-1,
                        pos_scale=batch["pixel_values"].shape[-1] // batch["cond_pixel_values"].shape[-1],
                    ).to(accelerator.device)

                # discard cond tokens that are entirely composed of zero values
                cond_seq_lens = [0 for _ in range(bsz)]
                pad_seq_lens = [0 for _ in range(bsz)]
                if args.model.is_use_cond_token:
                    if args.model.is_filter_cond_token:
                        cond_pixel_latents, cond_ids, cond_seq_lens, pad_seq_lens = Flux3RegionalPipeline.filter_cond_token(
                            batch["cond_pixel_values"],
                            cond_pixel_latents,
                            cond_ids,
                            vae_scale_factor=vae_scale_factor*2
                        )
                    else:
                        cond_seq_lens = [cond_pixel_latents.shape[1] for _ in range(bsz)]

                if args.trainer.offload:
                    # offload vae to CPU.
                    vae.cpu()

                noise = torch.randn_like(pixel_latents, device=accelerator.device, dtype=weight_dtype)
                # Sample a random timestep for each image
                u = compute_density_for_timestep_sampling(
                    weighting_scheme=args.model.weighting_scheme,
                    batch_size=bsz,
                    logit_mean=args.model.logit_mean,
                    logit_std=args.model.logit_std,
                    mode_scale=args.model.mode_scale,
                )
                indices = (u * noise_scheduler_copy.config.num_train_timesteps).long()
                timesteps = noise_scheduler_copy.timesteps[indices].to(device=pixel_latents.device)

                # Add noise according to flow matching.
                # zt = (1 - texp) * x + texp * z1
                sigmas = get_sigmas(timesteps, n_dim=pixel_latents.ndim, dtype=pixel_latents.dtype)
                noisy_model_input = (1.0 - sigmas) * pixel_latents + sigmas * noise

                # pack the latents. VAE 側で 2x2 パッチ化済みなので、ここは (B, C, h, w) -> (B, h*w, C) だけ
                latent_h, latent_w = noisy_model_input.shape[2], noisy_model_input.shape[3]
                packed_noisy_model_input = Flux3RegionalPipeline.pack_latents(noisy_model_input)

                # latent image ids for RoPE: (h*w, 4) = (0, 0.5h, 0.5w, 0)
                latent_image_ids = prepare_image_ids(latent_h, latent_w).to(accelerator.device)

                # FLUX.3 は guidance 埋め込みを持たない。学習は Base 版 (flow_map=False) なので target_timestep も無し
                if unwrap_flux_transformer.config.flow_map:
                    raise ValueError("蒸留版 (flow map) では学習しない。Base 版 (flux-3-image-klein-base) を指定する")

                # text encoding.
                text_encoding_pipeline = text_encoding_pipeline.to(accelerator.device)
                global_caption = batch["global_caption"]
                regional_captions = batch["regional_captions"]

                with torch.no_grad():
                    # 推論と同じ関数: チャットテンプレート -> 80 の倍数にパディング -> Qwen3-VL -> 連結。L は caption ごとに 0 から
                    prompt_embeds, txt_seq_lens, text_ids = text_encoding_pipeline.encode_all_prompt(
                        global_prompt=global_caption,
                        regional_prompts=regional_captions,
                        global_max_sequence_length=args.model.max_sequence_length,
                        regional_max_sequence_length=args.model.regional_max_sequence_length,
                        device=accelerator.device,
                        regional_text_position=args.model.regional_text_position,
                        text_pad_multiple=args.model.get("text_pad_multiple", TEXT_PAD_MULTIPLE),
                    )
                    prompt_embeds = prompt_embeds.to(weight_dtype)

                # prepare attention mask
                attention_mask, hard_attention_mask = Flux3RegionalPipeline.prepare_attention_mask(
                    attention_mask_method = args.model.attention_mask_method,
                    regional_labels = batch['label'],
                    txt_seq_lens = txt_seq_lens,
                    cond_seq_lens=cond_seq_lens,
                    pad_seq_lens = pad_seq_lens,
                    height=latent_h,
                    width=latent_w,
                    num_attention_heads=unwrap_flux_transformer.config.num_attention_heads,
                    dtype=weight_dtype,
                    device= accelerator.device,
                )

                joint_attention_kwargs = {}
                joint_attention_kwargs["attention_mask"] = attention_mask
                joint_attention_kwargs["hard_attention_mask"] = hard_attention_mask

                if args.trainer.offload:
                    text_encoding_pipeline = text_encoding_pipeline.to("cpu")

                # semcond: cond トークンごとの意味ベクトル (領域ごとの caption 埋め込みを、パッチ内の面積比で混ぜる)
                if use_semcond:
                    with torch.no_grad():
                        pooled = region_pooled_embeds(
                            text_encoding_pipeline, prompt_embeds, txt_seq_lens, regional_captions,
                            args.model.regional_max_sequence_length,
                        )
                        cond_h = batch["cond_pixel_values"].shape[-2] // (vae_scale_factor * 2)
                        cond_w = batch["cond_pixel_values"].shape[-1] // (vae_scale_factor * 2)
                        sem = cond_semantic(
                            pooled, batch["label"], cond_ids, cond_seq_lens, (cond_h, cond_w),
                            pos_scale=batch["pixel_values"].shape[-1] // batch["cond_pixel_values"].shape[-1],
                        )
                    unwrap_model(flux_transformer)._cond_sem = sem

                # Predict.
                model_pred = flux_transformer(
                    hidden_states = packed_noisy_model_input,
                    cond_hidden_states = cond_pixel_latents,
                    hard_attn_block_range = args.model.hard_attn_block_range,
                    encoder_hidden_states = prompt_embeds,
                    timestep = timesteps / 1000,
                    img_ids=latent_image_ids,
                    txt_ids=text_ids,
                    cond_ids=cond_ids,
                    joint_attention_kwargs=joint_attention_kwargs,
                    return_dict=False
                )[0]

                model_pred = unpack_latents(model_pred, latent_h, latent_w)

                # these weighting schemes use a uniform timestep sampling
                # and instead post-weight the loss
                weighting = compute_loss_weighting_for_sd3(weighting_scheme=args.model.weighting_scheme, sigmas=sigmas)

                # flow-matching loss
                target = noise - pixel_latents
                loss = torch.mean(
                    (weighting.float() * (model_pred.float() - target.float()) ** 2).reshape(target.shape[0], -1),
                    1,
                )
                loss = loss.mean()
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params_to_clip = flux_transformer.parameters()
                    accelerator.clip_grad_norm_(params_to_clip, args.optimizer.max_grad_norm)
                
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process or accelerator.distributed_type == DistributedType.DEEPSPEED:
                    if global_step % args.trainer.checkpointing_steps == 0:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.trainer.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.project.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.trainer.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.trainer.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.project.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(args.project.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

                if accelerator.is_main_process:
                    if args.trainer.num_validation_images >0 and global_step % args.trainer.validation_steps == 0:
                        image_logs = log_validation(
                            flux_transformer=flux_transformer,
                            text_encoding_pipeline=text_encoding_pipeline,
                            args=args,
                            val_dataloader=val_dataloader,
                            accelerator=accelerator,
                            weight_dtype=weight_dtype,
                            step=global_step,
                        )

            loss_logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**loss_logs)
            accelerator.log(loss_logs, step=global_step)

            if global_step >= args.trainer.max_train_steps:
                break

    accelerator.wait_for_everyone()
    accelerator.end_training()


if __name__ == "__main__":
    parser = get_parser()
    opt, unknown = parser.parse_known_args()
    unknown = [s.lstrip('-') for s in unknown]
    configs = [OmegaConf.load(cfg) for cfg in opt.base]
    cli = OmegaConf.from_dotlist(unknown)
    print('###### cli input training setup:  ######\n',cli)
    config = OmegaConf.merge(*configs, cli)
    
    if config.resolution % (16 * config.cond_scale_factor) != 0:
        raise ValueError(
            f"Image resolution {config.resolution} must be divisible by {16 * config.cond_scale_factor} "
            f"(16 * cond_scale_factor) to ensure proper feature map alignment in the model. "
            f"Please adjust either the resolution or cond_scale_factor."
        )
    
    main(config)