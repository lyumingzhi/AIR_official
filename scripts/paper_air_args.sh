#!/usr/bin/env bash
# Shared paper-style AIR arguments: AIA + ATI + RFA with the 8 surrogate FR models.
AIR_PAPER_ARGS=(
  --source_model simswap
  --relighting
  --total_mask
  --hard_constraint
  --lossType ensemble_wb_test
  --testType df
  --testAttackType fr
  --name_of_arcface_insight r18
  --name_of_arcface_insight r18
  --name_of_arcface_insight r34
  --name_of_arcface_insight r34
  --name_of_arcface_insight r50
  --name_of_arcface_insight r50
  --name_of_arcface_insight r100
  --name_of_arcface_insight r100
  --checkpoint_of_arcface_insight advDF/insightface/recognition/arcface_torch/checkpoint/ms1mv3_arcface_r18_fp16/backbone.pth
  --checkpoint_of_arcface_insight advDF/insightface/recognition/arcface_torch/checkpoint/glint360k_cosface_r18_fp16_0.1/backbone.pth
  --checkpoint_of_arcface_insight advDF/insightface/recognition/arcface_torch/checkpoint/ms1mv3_arcface_r34_fp16/backbone.pth
  --checkpoint_of_arcface_insight advDF/insightface/recognition/arcface_torch/checkpoint/glint360k_cosface_r34_fp16_0.1/backbone.pth
  --checkpoint_of_arcface_insight advDF/insightface/recognition/arcface_torch/checkpoint/ms1mv3_arcface_r50_fp16/backbone.pth
  --checkpoint_of_arcface_insight advDF/insightface/recognition/arcface_torch/checkpoint/glint360k_cosface_r50_fp16_0.1/backbone.pth
  --checkpoint_of_arcface_insight advDF/insightface/recognition/arcface_torch/checkpoint/glint360k_cosface_r100_fp16_0.1/backbone.pth
  --checkpoint_of_arcface_insight advDF/insightface/recognition/arcface_torch/checkpoint/ms1mv3_arcface_r100_fp16/backbone.pth
)
