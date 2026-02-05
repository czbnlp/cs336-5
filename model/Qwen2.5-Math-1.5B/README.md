---
base_model: Qwen/Qwen2.5-1.5B
language:
- en
pipeline_tag: text-generation
library_name: transformers
license: apache-2.0
license_link: https://huggingface.co/Qwen/Qwen2.5-Math-1.5B/blob/main/LICENSE
---

# Qwen2.5-Math-1.5B

> [!Warning]
> <div align="center">
> <b>
> 🚨 Qwen2.5-Math mainly supports solving English and Chinese math problems through CoT and TIR. We do not recommend using this series of models for other tasks.
> </b>
> </div>

## Introduction

In August 2024, we released the first series of mathematical LLMs - [Qwen2-Math](https://qwenlm.github.io/blog/qwen2-math/) - of our Qwen family. A month later, we have upgraded it and open-sourced **Qwen2.5-Math** series, including base models **Qwen2.5-Math-1.5B/7B/72B**, instruction-tuned models **Qwen2.5-Math-1.5B/7B/72B-Instruct**, and mathematical reward model **Qwen2.5-Math-RM-72B**. 
                                                                              
Unlike Qwen2-Math series which only supports using Chain-of-Thught (CoT) to solve English math problems, Qwen2.5-Math series is expanded to support using both CoT and Tool-integrated Reasoning (TIR) to solve math problems in both Chinese and English. The Qwen2.5-Math series models have achieved significant performance improvements compared to the Qwen2-Math series models on the Chinese and English mathematics benchmarks with CoT. 
![](http://qianwen-res.oss-accelerate-overseas.aliyuncs.com/Qwen2.5/qwen2.5-math-pipeline.jpeg)

While CoT plays a vital role in enhancing the reasoning capabilities of LLMs, it faces challenges in achieving computational accuracy and handling complex mathematical or algorithmic reasoning tasks, such as finding the roots of a quadratic equation or computing the eigenvalues of a matrix. TIR can further improve the model's proficiency in precise computation, symbolic manipulation, and algorithmic manipulation. Qwen2.5-Math-1.5B/7B/72B-Instruct achieve 79.7, 85.3, and 87.8 respectively on the MATH benchmark using TIR. 

## Model Details


For more details, please refer to our [blog post](https://qwenlm.github.io/blog/qwen2.5-math/) and [GitHub repo](https://github.com/QwenLM/Qwen2.5-Math).


## Requirements
* `transformers>=4.37.0` for Qwen2.5-Math models. The latest version is recommended.

> [!Warning]
> <div align="center">
> <b>
> 🚨 This is a must because <code>transformers</code> integrated Qwen2 codes since <code>4.37.0</code>.
> </b>
> </div>

For requirements on GPU memory and the respective throughput, see similar results of Qwen2 [here](https://qwen.readthedocs.io/en/latest/benchmark/speed_benchmark.html).

## Quick Start

> [!Important]
>
> **Qwen2.5-Math-1.5B-Instruct** is an instruction model for chatting;
>
> **Qwen2.5-Math-1.5B** is a base model typically used for completion and few-shot inference, serving as a better starting point for fine-tuning.

## Citation

If you find our work helpful, feel free to give us a citation.

```
@article{yang2024qwen25mathtechnicalreportmathematical,
  title={Qwen2.5-Math Technical Report: Toward Mathematical Expert Model via Self-Improvement}, 
  author={An Yang and Beichen Zhang and Binyuan Hui and Bofei Gao and Bowen Yu and Chengpeng Li and Dayiheng Liu and Jianhong Tu and Jingren Zhou and Junyang Lin and Keming Lu and Mingfeng Xue and Runji Lin and Tianyu Liu and Xingzhang Ren and Zhenru Zhang},
  journal={arXiv preprint arXiv:2409.12122},
  year={2024}
}
```