# RepairLLaMA: Efficient Representations and Fine-Tuned Adapters for Program Repair

If you use RepairLLaMA in academic research, please cite "[RepairLLaMA: Efficient Representations and Fine-Tuned Adapters for Program Repair](http://arxiv.org/abs/2312.15698)", Technical report, arXiv 2312.15698, 2023. 

```bibtex
@techreport{repairllama2023,
  title={RepairLLaMA: Efficient Representations and Fine-Tuned Adapters for Program Repair},
  author={Silva, Andr{\'e} and Fang, Sen and Monperrus, Martin},
  url = {http://arxiv.org/abs/2312.15698},
  number = {2312.15698},
  institution = {arXiv},
}
```

This repository contains the code, model, and results to replicate the paper "RepairLLaMA: Efficient Representations and Fine-Tuned Adapters for Program Repair"

It is structured as follows:
- [repairllama-lora](repairllama-lora) contains the RepairLLaMA low-rank adaptation of CodeLLaMA-7B, called "repair adapter"
- [results](results) contains all generated patches for Defects4J and HumanEval-Java by all models (incl. full fine-tuning, lora, and code representations)
- [src](src) contains the training and inference scripts, and scripts to generate datasets for different input-output representations (IRxOR)
- [example](example) contains an example notebook explaining how to load and prompt the RepairLLaMA model
- [benchmarks](benchmarks) contains the datasets for different input-output representations (IRxOR)

## Models

All fine-tuned models are available on HuggingFace, here are specific links:

- IR1xOR1: [https://huggingface.co/ASSERT-KTH/RepairLLaMA-IR1-OR1](https://huggingface.co/ASSERT-KTH/RepairLLaMA-IR1-OR1)
- IR1xOR3: [https://huggingface.co/ASSERT-KTH/RepairLLaMA-IR1-OR3](https://huggingface.co/ASSERT-KTH/RepairLLaMA-IR1-OR3)
- IR1xOR3: [https://huggingface.co/ASSERT-KTH/RepairLLaMA-IR1-OR4](https://huggingface.co/ASSERT-KTH/RepairLLaMA-IR1-OR4)
- IR2xOR2: [https://huggingface.co/ASSERT-KTH/RepairLLaMA-IR2-OR2](https://huggingface.co/ASSERT-KTH/RepairLLaMA-IR2-OR2)
- IR3xOR2: [https://huggingface.co/ASSERT-KTH/RepairLLaMA-IR3-OR2](https://huggingface.co/ASSERT-KTH/RepairLLaMA-IR3-OR2)

## Datasets

The processed fine-tuning datasets are made available on HuggingFace at [https://huggingface.co/datasets/ASSERT-KTH/repairllama-datasets](https://huggingface.co/datasets/ASSERT-KTH/repairllama-datasets).
It contains the datasets used for training the RepairLLaMA models, one subset per input/output representation pair.
To get the 30k..50k datasets we did further filtering based on the token length of input + output pairs being less than 1024 tokens.

If it interests you, you can also find these on our HuggingFace org:
  - Megadiff (original dataset, in HF format): [https://huggingface.co/datasets/ASSERT-KTH/megadiff](https://huggingface.co/datasets/ASSERT-KTH/megadiff)
  - Megadiff Single-Function (single-function diffs only, with buggy and fixed functions extracted from it): [https://huggingface.co/datasets/ASSERT-KTH/megadiff-single-function](https://huggingface.co/datasets/ASSERT-KTH/megadiff-single-function)

## Benchmarks

The evaluation benchmarks are [Defects4J v2](https://github.com/rjust/defects4j), [HumanEval-Java](https://github.com/ASSERT-KTH/human-eval-java), and [GitBug-Java](https://github.com/gitbugactions/gitbug-java).

We focus on single-function bugs (i.e. bugs whose developer patch exclusively changes one function):
  - Defects4J contains 488 single-function bugs: [defects4j_sf.txt](results/benchmarks/defects4j_sf.txt)
  - HumanEval-Java contains 162 single-function bugs: [humanevaljava_sf.txt](results/benchmarks/humanevaljava_sf.txt)
  - GitBug-Java contains 90 single-functions bugs: [gitbugjava_sf.txt](results/benchmarks/gitbugjava_sf.txt)

Note that the original HumanEval-Java contains a [duplicate bug](https://github.com/lin-tan/clm/issues/2).
