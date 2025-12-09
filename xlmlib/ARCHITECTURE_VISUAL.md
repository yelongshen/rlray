# Visual Architecture Guide

## System Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    MINIMAL PRETRAINING                       │
│                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │     Data     │  │    Tensor    │  │   Pipeline   │      │
│  │   Parallel   │  │   Parallel   │  │   Parallel   │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐   │
│  │           MinimalTransformer Model                   │   │
│  │  ┌────────┐  ┌────────┐  ┌────────┐  ┌────────┐    │   │
│  │  │ Embed  │→ │ Layer  │→ │  ...   │→ │ LM Head│    │   │
│  │  └────────┘  └────────┘  └────────┘  └────────┘    │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

## Data Parallel (DP)

```
┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐
│   GPU 0     │  │   GPU 1     │  │   GPU 2     │  │   GPU 3     │
│             │  │             │  │             │  │             │
│   Model     │  │   Model     │  │   Model     │  │   Model     │
│  (replica)  │  │  (replica)  │  │  (replica)  │  │  (replica)  │
│             │  │             │  │             │  │             │
│   Batch A   │  │   Batch B   │  │   Batch C   │  │   Batch D   │
└─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘
       │                │                │                │
       └────────────────┴────────────────┴────────────────┘
                        │
                   AllReduce
                   Gradients
```

**Key Points:**
- Each GPU has full model replica
- Different data batches on each GPU
- Gradients averaged via AllReduce
- Good when: Model fits on single GPU

## Tensor Parallel (TP)

```
┌───────────────────────────────────────────────────────────┐
│                     Single Layer                          │
│                                                           │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐      │
│  │   GPU 0     │  │   GPU 1     │  │   GPU 2     │      │
│  │             │  │             │  │             │      │
│  │  Heads 0-3  │  │  Heads 4-7  │  │  Heads 8-11 │      │
│  │             │  │             │  │             │      │
│  │  MLP[0:512] │  │ MLP[512:1K] │  │ MLP[1K:1.5K]│      │
│  └─────────────┘  └─────────────┘  └─────────────┘      │
│         │                │                │               │
│         └────────────────┴────────────────┘               │
│                      AllReduce                            │
└───────────────────────────────────────────────────────────┘

Attention:
  Q, K, V → Column Parallel (split heads)
  O       → Row Parallel (gather then reduce)

MLP:
  Gate, Up → Column Parallel (split features)
  Down     → Row Parallel (gather then reduce)
```

**Key Points:**
- Model split across GPUs
- Each GPU has portion of weights
- AllReduce after each layer
- Good when: Model too large for single GPU
- Requires: High bandwidth (NVLink)

## Pipeline Parallel (PP)

```
Time →

GPU 0     GPU 1     GPU 2     GPU 3
(Stage0)  (Stage1)  (Stage2)  (Stage3)
┌──────┐
│ MB 0 │
├──────┤→ ┌──────┐
│ MB 1 │  │ MB 0 │
├──────┤→ ├──────┤→ ┌──────┐
│ MB 2 │  │ MB 1 │  │ MB 0 │
├──────┤→ ├──────┤→ ├──────┤→ ┌──────┐
│ MB 3 │  │ MB 2 │  │ MB 1 │  │ MB 0 │
└──────┘→ ├──────┤→ ├──────┤→ ├──────┤
   ↑      │ MB 3 │  │ MB 2 │  │ MB 1 │
   │      └──────┘→ ├──────┤→ ├──────┤
Backward     ↑      │ MB 3 │  │ MB 2 │
Pass         │      └──────┘→ ├──────┤
             │         ↑      │ MB 3 │
          Backward     │      └──────┘
          Pass      Backward     ↑
                    Pass      Backward
                              Pass

Stage 0: Layers  0-11
Stage 1: Layers 12-23
Stage 2: Layers 24-35
Stage 3: Layers 36-47

MB = Microbatch
```

**Key Points:**
- Layers split across GPUs
- Microbatches flow through pipeline
- 1F1B: Alternate forward/backward
- Good when: Very deep models
- Challenge: Pipeline bubbles

## Combined: DP × TP × PP

```
Example: 16 GPUs = 2 (DP) × 2 (TP) × 4 (PP)

Data Parallel Group 0:
┌─────────────────────────────────────────────┐
│  PP Stage 0      PP Stage 1                 │
│  ┌─────┬─────┐  ┌─────┬─────┐              │
│  │GPU0 │GPU1 │→ │GPU2 │GPU3 │→ ...         │
│  └─────┴─────┘  └─────┴─────┘              │
│    TP Group       TP Group                  │
└─────────────────────────────────────────────┘

Data Parallel Group 1:
┌─────────────────────────────────────────────┐
│  PP Stage 0      PP Stage 1                 │
│  ┌─────┬─────┐  ┌─────┬─────┐              │
│  │GPU8 │GPU9 │→ │GPU10│GPU11│→ ...         │
│  └─────┴─────┘  └─────┴─────┘              │
│    TP Group       TP Group                  │
└─────────────────────────────────────────────┘

Communication:
- TP: AllReduce within TP group (high bandwidth)
- PP: P2P between stages (lower bandwidth)
- DP: AllReduce across DP groups (after backward)
```

## Process Group Structure

```
World: All 16 GPUs
├─ DP Group 0: [0, 1, 2, 3, 4, 5, 6, 7]
│  ├─ PP Stage 0
│  │  └─ TP Group: [0, 1]
│  ├─ PP Stage 1
│  │  └─ TP Group: [2, 3]
│  ├─ PP Stage 2
│  │  └─ TP Group: [4, 5]
│  └─ PP Stage 3
│     └─ TP Group: [6, 7]
│
└─ DP Group 1: [8, 9, 10, 11, 12, 13, 14, 15]
   ├─ PP Stage 0
   │  └─ TP Group: [8, 9]
   ├─ PP Stage 1
   │  └─ TP Group: [10, 11]
   ├─ PP Stage 2
   │  └─ TP Group: [12, 13]
   └─ PP Stage 3
      └─ TP Group: [14, 15]
```

## Transformer Layer Detail

```
┌────────────────────────────────────────┐
│         Transformer Block              │
│                                        │
│  Input                                 │
│    ↓                                   │
│  ┌─────────────┐                       │
│  │  LayerNorm  │                       │
│  └─────────────┘                       │
│    ↓                                   │
│  ┌─────────────────────────────────┐   │
│  │      Multi-Head Attention       │   │
│  │  ┌───────────────────────────┐  │   │
│  │  │ Q,K,V (Column Parallel)   │  │   │  ← Split across TP
│  │  │ Attention                 │  │   │
│  │  │ O (Row Parallel)          │  │   │  ← Gather + Reduce
│  │  └───────────────────────────┘  │   │
│  └─────────────────────────────────┘   │
│    ↓                                   │
│  Residual +                            │
│    ↓                                   │
│  ┌─────────────┐                       │
│  │  LayerNorm  │                       │
│  └─────────────┘                       │
│    ↓                                   │
│  ┌─────────────────────────────────┐   │
│  │           MLP (SwiGLU)          │   │
│  │  ┌───────────────────────────┐  │   │
│  │  │ Gate, Up (Column Parallel)│  │   │  ← Split across TP
│  │  │ SiLU                      │  │   │
│  │  │ Down (Row Parallel)       │  │   │  ← Gather + Reduce
│  │  └───────────────────────────┘  │   │
│  └─────────────────────────────────┘   │
│    ↓                                   │
│  Residual +                            │
│    ↓                                   │
│  Output                                │
└────────────────────────────────────────┘
```

## Memory Distribution

```
Single GPU (No Parallelism):
┌─────────────────────────────┐
│  GPU 0                      │
│  ┌────────────────────────┐ │
│  │  Model (100%)          │ │  ← Full model
│  │  Optimizer (100%)      │ │  ← Full optimizer
│  │  Gradients (100%)      │ │  ← Full gradients
│  │  Activations           │ │
│  └────────────────────────┘ │
└─────────────────────────────┘

Tensor Parallel (4 GPUs):
┌───────┐ ┌───────┐ ┌───────┐ ┌───────┐
│ GPU 0 │ │ GPU 1 │ │ GPU 2 │ │ GPU 3 │
│ Model │ │ Model │ │ Model │ │ Model │
│  25%  │ │  25%  │ │  25%  │ │  25%  │  ← Split weights
│ Opt   │ │ Opt   │ │ Opt   │ │ Opt   │
│  25%  │ │  25%  │ │  25%  │ │  25%  │  ← Split optimizer
│ Full  │ │ Full  │ │ Full  │ │ Full  │
│ Acts  │ │ Acts  │ │ Acts  │ │ Acts  │  ← Activations
└───────┘ └───────┘ └───────┘ └───────┘

Pipeline Parallel (4 GPUs):
┌───────┐ ┌───────┐ ┌───────┐ ┌───────┐
│ GPU 0 │ │ GPU 1 │ │ GPU 2 │ │ GPU 3 │
│ Model │ │ Model │ │ Model │ │ Model │
│  25%  │ │  25%  │ │  25%  │ │  25%  │  ← Split layers
│ Opt   │ │ Opt   │ │ Opt   │ │ Opt   │
│  25%  │ │  25%  │ │  25%  │ │  25%  │  ← Split optimizer
│ Small │ │ Small │ │ Small │ │ Small │
│ Acts  │ │ Acts  │ │ Acts  │ │ Acts  │  ← Less activations
└───────┘ └───────┘ └───────┘ └───────┘
```

## Training Flow

```
┌─────────────────────────────────────────────────────────┐
│                    Training Loop                        │
└─────────────────────────────────────────────────────────┘
                         │
        ┌────────────────┴────────────────┐
        │                                 │
        ▼                                 ▼
┌──────────────┐                 ┌──────────────┐
│  Load Batch  │                 │    Model     │
│              │                 │              │
│  DP: Split   │                 │  TP: Split   │
│  across DPs  │                 │  weights     │
└──────────────┘                 └──────────────┘
        │                                 │
        └────────────────┬────────────────┘
                         │
                         ▼
        ┌─────────────────────────────────┐
        │    Forward Pass                 │
        │                                 │
        │  PP: Stage by stage             │
        │  TP: AllReduce per layer        │
        └─────────────────────────────────┘
                         │
                         ▼
        ┌─────────────────────────────────┐
        │    Compute Loss                 │
        │    (only on last PP stage)      │
        └─────────────────────────────────┘
                         │
                         ▼
        ┌─────────────────────────────────┐
        │    Backward Pass                │
        │                                 │
        │  PP: Stage by stage (reverse)   │
        │  TP: AllReduce per layer        │
        └─────────────────────────────────┘
                         │
                         ▼
        ┌─────────────────────────────────┐
        │    Gradient AllReduce           │
        │    (across DP groups)           │
        └─────────────────────────────────┘
                         │
                         ▼
        ┌─────────────────────────────────┐
        │    Optimizer Step               │
        │    (update local weights)       │
        └─────────────────────────────────┘
                         │
                         ▼
                    Next Iteration
```

## Decision Tree

```
┌─────────────────┐
│  Need to scale? │
└────────┬────────┘
         │
    ┌────┴────┐
    │  Yes    │  No → Single GPU
    └────┬────┘
         │
    ┌────┴──────────────────┐
    │ Model fits on 1 GPU?  │
    └────┬──────────────────┘
         │
    ┌────┴────┐
    │  Yes    │  No → TP or PP
    └────┬────┘              │
         │                   │
    Data Parallel        ┌───┴────────┐
    (simplest)           │ Wide model?│
                         └───┬────────┘
                             │
                        ┌────┴────┐
                        │  Yes    │  No → Pipeline Parallel
                        └────┬────┘
                             │
                        Tensor Parallel
```

## File Dependencies

```
parallel_initialization.py
    ↓ (used by)
minimal_model.py
    ↓ (used by)
pipeline_parallel.py
    ↓ (used by)
minimal_pretrain.py ← Main entry point
    ↑ (configured by)
pretrain_config.py

Supporting files:
- launch_pretrain.sh/.ps1
- test_minimal_pretrain.py
- examples_pretrain.py
- README_PRETRAIN.md
```

## Key Classes

```
minimal_model.py:
  ├─ MinimalTransformer (main model)
  ├─ TransformerBlock (one layer)
  ├─ Attention (multi-head attention)
  ├─ MLP (feedforward)
  ├─ ColumnParallelLinear (TP)
  ├─ RowParallelLinear (TP)
  └─ VocabParallelEmbedding (TP)

pipeline_parallel.py:
  ├─ PipelineParallel (wrapper)
  ├─ PipelineStage (stage management)
  ├─ GPipeSchedule
  └─ OneFOneBSchedule
```

This visual guide provides a comprehensive overview of the architecture and parallelism strategies!
