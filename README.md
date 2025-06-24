# Awesome-LLM-Inference-On-Android

[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://github.com/zhouwg/Awesome-LLM-Inference-On-Android)
[![Last Commit](https://img.shields.io/github/last-commit/zhouwg/Awesome-LLM-Inference-On-Android)](https://github.com/zhouwg/Awesome-LLM-Inference-On-Android)
[![Contribution Welcome](https://img.shields.io/badge/Contributions-welcome-blue)]()


##  Introduction

Focus on LLM inference on Android phone, espcially Android phone equipped with Qualcomm Snapdragon high-end mobile SoC, such as Snapdragon 8Gen3, 8Elite, 8Elite2....

Maintain an up-to-date Github repo to track the latest development of foundatinonal technologies, benchmarks, future directions in this rapidly evolving field.


##  Table of Contents

- [Awesome-LLM-Inference-On-Android](#awesome-llm-inference-on-android)
  - [Part 1: Research](#part-1-research)
  - [Part 2: On-Device Inference framework](#part-2-on-device-inference-framework)
  - [Part 3: Hardware acceleration](#part-3-hardware-acceleration)
  - [Part 4: Android APPs](#part-4-android-apps)
  - [Part 5: References](#part-5-references)


## Part 1: Research
* HeteroLLM: Accelerating Large Language Model Inference on Mobile SoCs platform with Heterogeneous AI Accelerators  [[Paper]](https://arxiv.org/abs/2501.14794) ![](https://img.shields.io/badge/arXiv-2025.01-red)
* LLM prefilling with mllm-NPU [[Paper]](https://arxiv.org/abs/2407.05858v1) ![](https://img.shields.io/badge/arXiv-2024.07-red) , https://github.com/UbiquitousLearning/mllm
* PowerInfer-2: Fast Large Language Model Inference on a Smartphone  [[Paper]](https://arxiv.org/abs/2406.06282) ![](https://img.shields.io/badge/arXiv-2024.06-red) , https://github.com/SJTU-IPADS/PowerInfer
* T-MAC: CPU Renaissance via Table Lookup for Low-Bit LLM Deployment on Edge  [[Paper]](https://arxiv.org/abs/2407.00088v1) ![](https://img.shields.io/badge/arXiv-2024.06-red) , https://github.com/microsoft/T-MAC

## Part 2: On-Device Inference framework

* https://github.com/ggml-org/llama.cpp
* https://github.com/ikawrakow/ik_llama.cpp ([ikawrakow](https://github.com/ikawrakow) did [an unique and important contribution](https://github.com/ggml-org/llama.cpp/pull/1684) in upstream llama.cpp and he is still focusing on quantization techniques in llama.cpp)
* https://github.com/mlc-ai/mlc-llm
* https://github.com/alibaba/MNN
* https://github.com/Tencent/ncnn
* https://github.com/XiaoMi/mace
* https://github.com/OpenPPL/ppl.nn
* https://github.com/pytorch/executorch
* https://github.com/google-ai-edge/LiteRT

## Part 3: Hardware acceleration

QNN solution, aka End-to-End solution, which is similar to the Qualcomm's official AI-Hub solution.

llama.cpp solution: a specified ggml backend for llama.cpp on Android.

* QNN solution: https://github.com/SJTU-IPADS/PowerInfer
* QNN solution: https://github.com/UbiquitousLearning/mllm
* QNN solution: https://github.com/MollySophia/rwkv-qualcomm
* llama.cpp solution: https://github.com/zhouwg/ggml-hexagon, the first original llama.cpp solution which launched on 03/2024(the initial version was reverse engineered from [Qualcomm's codes in executorch](https://github.com/pytorch/executorch/tree/main/backends/qualcomm))
* llama.cpp solution: https://github.com/chraac/llama.cpp (hard-forked from zhouwg's initial version)
* RKNN solution: https://github.com/airockchip/rknn-llm

## Part 4: Android APPs
* https://github.com/a-ghorbani/pocketpal-ai
* https://github.com/Vali-98/ChatterUI
* https://github.com/shubham0204/SmolChat-Android
* https://github.com/kantv-ai/kantv

## Part 5: References
* Qualcomm: https://docs.qualcomm.com/bundle/publicresource/topics/80-63442-50/introduction.html
* MTK: https://neuropilot.mediatek.com/resources/public/npexpress/en/docs/npexpress
* Rockchip: https://github.com/airockchip/rknn-toolkit2/
