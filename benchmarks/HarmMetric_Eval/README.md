# HarmMetric Eval

This is the official repository of ***HarmMetric Eval: Benchmarking Metrics and Judges for LLM Harmfulness Assessment***.

The file `benchmark_codes/scoring.py` contains the core code for scoring in our benchmark.

This repository contains our benchmark dataset, training data, and test data in the `data` folder, provided as `dataset.jsonl`, `train_data.jsonl`, and `test_data.jsonl`, respectively. You can also download these files from [here](https://huggingface.co/datasets/anonymous-review-anonymous/HarmMetric_Eval).

Our trained classifier is available at [here](https://huggingface.co/anonymous-review-anonymous/HarmClassifier).


## Abstract
The potential for large language models (LLMs) to generate harmful content poses a significant safety risk in their deployment. To address and assess this risk, the community has developed numerous harmfulness evaluation metrics and judges. However, the lack of a systematic benchmark for evaluating these metrics and judges undermines the credibility and consistency of LLM safety assessments. 
To bridge this gap, we introduce *HarmMetric Eval*, a comprehensive benchmark designed to support both overall and fine-grained evaluation of harmfulness metrics and judges. 
In *HarmMetric Eval*, we build a high-quality dataset of representative harmful prompts paired with highly diverse harmful model responses and non-harmful counterparts across multiple categories. We also propose a flexible scoring mechanism that rewards the metrics for correctly ranking harmful responses above non-harmful ones, which is applicable to almost all existing metrics and judges with varying output formats and scoring scales.
Using *HarmMetric Eval*, we uncover a surprising finding by extensive experiments: Conventional reference-based metrics such as ROUGE and METEOR can outperform existing LLM-based judges in fine-grained harmfulness evaluation, challenging prevailing assumptions about LLMs’ superiority in this domain. To reveal the reasons behind this finding, we provide a fine-grained analysis to explain the limitations of LLM-based judges on rating irrelevant or useless responses.
Furthermore, we build a new harmfulness judge by incorporating the fine-grained criteria into its prompt template and leverage reference-based metrics to fine-tune its base LLM. The resulting judge demonstrates superior performance than all existing metrics and judges in evaluating harmful responses.

![alt text](pipeline.png)


## Quick Start

The following is only the evaluation process for **GCG_Eval** and **AutoDAN_Eval**.  

You can modify the `eval_list` at the beginning of `metrics_codes/eval_with_metrics.py` to select the **harmfulness metric** you wish to evaluate, or use your own metric. However, remember to update the model paths or OpenAI API keys for the corresponding methods in `metrics_codes/curr_metrics.py` to your own.  

You can build environment for all metrics by following the step below, except for **DAN_Eval**, which requires downgrading `transformers` to **4.27.1**:  

```bash
pip install -r requirements_evaluation.txt
```

If you only want to try **GCG_Eval** and **AutoDAN_Eval** and score them using *HarmMetric Eval*, the following command is sufficient to set up the required environment:

```bash
pip install pandas==2.2.3
```

### Experimental replication

Follow the steps below, and you will find the detailed evaluation results of **GCG_Eval** and **AutoDAN_Eval** on our dataset in the `temp_results` folder.

```bash
cd metrics_codes
python eval_with_metrics.py
cd ..
```

### Scoring metrics

Follow the steps below, and you will be able to see the **overall** and **fine-grained effectiveness scores** of **GCG_Eval** and **AutoDAN_Eval** on our benchmark in the `results` folder.

```bash
cd benchmark_codes
python scoring.py --metric all
cd ..
```

## Replicate the training process of *HarmClassifier*

If you would like to replicate the training process of HarmClassifier, first set up the environment by following the steps below:

```bash
pip install -r requirements_train.txt
```

Next, update the path to **Qwen2.5-7B-Instruct** in `train_codes/train.py` to point to your local model directory. Then run the following commands to start training:

```bash
cd train_codes
python train.py
```

After the training process completes, you will obtain the LoRA adapter for HarmClassifier.
