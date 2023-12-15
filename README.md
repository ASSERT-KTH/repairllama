# RepairLLaMA: Efficient Representations and Fine-Tuned Adapters for Program Repair

This repository contains the code, model, and results to replicate the paper "RepairLLaMA: Efficient Representations and Fine-Tuned Adapters for Program Repair"

It is structured as follows:
- [repair-lora](repair-lora) contains the RepairLLaMA low-rank adaptation of CodeLLaMA-7B, called "repair adapter"
- [results](results) contains all generated patches for Defects4J and HumanEval-Java by all models (incl. full fine-tuning, lora, and code representations)
- [src](src) contains the training and inference scripts
- [tool](tool) contains a dockerize version of RepairLLaMA for easy deployment
