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
- [repair-lora](repair-lora) contains the RepairLLaMA low-rank adaptation of CodeLLaMA-7B, called "repair adapter"
- [results](results) contains all generated patches for Defects4J and HumanEval-Java by all models (incl. full fine-tuning, lora, and code representations)
- [src](src) contains the training and inference scripts
- [example](example) contains an example notebook explaining how to load and prompt the RepairLLaMA model

## Models

All fine-tuned models will soon be made available on HuggingFace

## Datasets

The processed fine-tuning datasets are made available on HuggingFace at [https://huggingface.co/datasets/ASSERT-KTH/repairllama-datasets](https://huggingface.co/datasets/ASSERT-KTH/repairllama-datasets).
It contains the datasets used for training the RepairLLaMA models, one subset per input/output representation pair.
To get the 30k..50k datasets we did further filtering based on the token length of input + output pairs being less than 1024 tokens.

If it interests you, you can also find these on our HuggingFace org:
  - Megadiff (original dataset, in HF format): [https://huggingface.co/datasets/ASSERT-KTH/megadiff](https://huggingface.co/datasets/ASSERT-KTH/megadiff)
  - Megadiff Single-Function (single-function diffs only, with buggy and fixed functions extracted from it): [https://huggingface.co/datasets/ASSERT-KTH/megadiff-single-function](https://huggingface.co/datasets/ASSERT-KTH/megadiff-single-function)

## Benchmarks

The evaluation benchmarks are [Defects4J v2](https://github.com/rjust/defects4j) and [HumanEval-Java](https://github.com/ASSERT-KTH/human-eval-java).

We focus on single-function bugs (i.e. bugs whose developer patch exclusively changes one function):
  - Defects4J contains 488 single-function bugs: [defects4j_sf.txt](results/benchmarks/defects4j_sf.txt)
  - HumanEval-Java contains 162 single-function bugs: [humaneval_java.txt](results/benchmarks/humanevaljava_sf.txt)

Note that the original HumanEval-Java contains a [duplicate bug](https://github.com/lin-tan/clm/issues/2).
