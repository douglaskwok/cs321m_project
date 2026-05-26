import json
import time
import curr_metrics
import os

# ============================================================================
# Metric name mapping: Display name -> Function name in curr_metrics
# ============================================================================
METRIC_MAPPING = {
    "GCG_Eval": "GCG_eval",
    "AutoDAN_Eval": "AutoDAN_eval",
    "DAN_Eval": "Shen_eval",  # Note: may need to change transformers version

    "F1_Score": "squad_f1_score",
    "Cosine_Similarity": "cos_sim",
    "BERT_Similarity": "BERT_sim",
    "METEOR": "meteor",
    "BLEU": "bleu",
    
    "ROUGE-1-Fmeasure-w/Stemmer": "rouge1",
    "ROUGE-2-Fmeasure-w/Stemmer": "rouge2",
    "ROUGE-L-Fmeasure-w/Stemmer": "rougeL",
    "ROUGE-Lsum-Fmeasure-w/Stemmer": "rougeLsum",
    "ROUGE-1-Fmeasure-w/oStemmer": "rouge1-nostemmed",
    "ROUGE-2-Fmeasure-w/oStemmer": "rouge2-nostemmed",
    "ROUGE-L-Fmeasure-w/oStemmer": "rougeL-nostemmed",
    "ROUGE-Lsum-Fmeasure-w/oStemmer": "rougeLsum-nostemmed",
    "ROUGE-1-Recall-w/Stemmer": "rouge1-recall",
    "ROUGE-2-Recall-w/Stemmer": "rouge2-recall",
    "ROUGE-L-Recall-w/Stemmer": "rougeL-recall",
    "ROUGE-Lsum-Recall-w/Stemmer": "rougeLsum-recall",
    "ROUGE-1-Recall-w/oStemmer": "rouge1-recall-nostemmed",
    "ROUGE-2-Recall-w/oStemmer": "rouge2-recall-nostemmed",
    "ROUGE-L-Recall-w/oStemmer": "rougeL-recall-nostemmed",
    "ROUGE-Lsum-Recall-w/oStemmer": "rougeLsum-recall-nostemmed",
    "ROUGE-1-Precision-w/Stemmer": "rouge1-precision",
    "ROUGE-2-Precision-w/Stemmer": "rouge2-precision",
    "ROUGE-L-Precision-w/Stemmer": "rougeL-precision",
    "ROUGE-Lsum-Precision-w/Stemmer": "rougeLsum-precision",
    "ROUGE-1-Precision-w/oStemmer": "rouge1-precision-nostemmed",
    "ROUGE-2-Precision-w/oStemmer": "rouge2-precision-nostemmed",
    "ROUGE-L-Precision-w/oStemmer": "rougeL-precision-nostemmed",
    "ROUGE-Lsum-Precision-w/oStemmer": "rougeLsum-precision-nostemmed",
    
    "JailbreakBench_Eval": "JbB_eval",
    "PAIR_Eval": "PAIR_eval",
    "Fine-tuning_Eval": "Qi_eval",
    "StrongReject_Eval": "strongreject_eval",
    "HarmJudge": "new_prompt_eval",
    
    "LlamaGuard": "llama_guard_1_eval",
    "LlamaGuard-4": "llama_guard_4_eval",
    "WildGuard": "wildguard_eval",
    "ShieldGemma": "shieldgemma_eval",
    "HarmBench_Eval": "HarmBench_eval",
    "GPTFuzzer_Eval": "GPTF_eval",
    "HarmClassifier": "qwen_harmfulness_classifier",
    "Qwen3.5-0.8B-HarmJudge": "qwen35_08b_harmjudge",
}

# ============================================================================
# Configuration: Select metrics to evaluate
# ============================================================================
eval_list = [
    "Qwen3.5-0.8B-HarmJudge",
    # "GCG_Eval",
    # "AutoDAN_Eval",
    # "DAN_Eval",
    
    # "F1_Score",
    # "Cosine_Similarity",
    # "BERT_Similarity",
    # "METEOR",
    # "BLEU",
    
    # "ROUGE-1-Fmeasure-w/Stemmer",
    # "ROUGE-2-Fmeasure-w/Stemmer",
    # "ROUGE-L-Fmeasure-w/Stemmer",
    # "ROUGE-Lsum-Fmeasure-w/Stemmer",
    # "ROUGE-1-Fmeasure-w/oStemmer",
    # "ROUGE-2-Fmeasure-w/oStemmer",
    # "ROUGE-L-Fmeasure-w/oStemmer",
    # "ROUGE-Lsum-Fmeasure-w/oStemmer",
    # "ROUGE-1-Recall-w/Stemmer",
    # "ROUGE-2-Recall-w/Stemmer",
    # "ROUGE-L-Recall-w/Stemmer",
    # "ROUGE-Lsum-Recall-w/Stemmer",
    # "ROUGE-1-Recall-w/oStemmer",
    # "ROUGE-2-Recall-w/oStemmer",
    # "ROUGE-L-Recall-w/oStemmer",
    # "ROUGE-Lsum-Recall-w/oStemmer",
    # "ROUGE-1-Precision-w/Stemmer",
    # "ROUGE-2-Precision-w/Stemmer",
    # "ROUGE-L-Precision-w/Stemmer",
    # "ROUGE-Lsum-Precision-w/Stemmer",
    # "ROUGE-1-Precision-w/oStemmer",
    # "ROUGE-2-Precision-w/oStemmer",
    # "ROUGE-L-Precision-w/oStemmer",
    # "ROUGE-Lsum-Precision-w/oStemmer",
    
    # "JailbreakBench_Eval",
    # "PAIR_Eval",
    # "Fine-tuning_Eval",
    # "StrongReject_Eval",
    # "HarmJudge",
    
    # "LlamaGuard",
    # "LlamaGuard-4",
    # "WildGuard",
    # "ShieldGemma",
    # "HarmBench_Eval",
    # "GPTFuzzer_Eval",
    # "HarmClassifier",
]

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate HarmMetric metrics/judges")
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=None,
        help="Metric display names from METRIC_MAPPING. Defaults to eval_list.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional max number of non-reference responses to evaluate per metric.",
    )
    parser.add_argument(
        "--prompt-file",
        type=str,
        default=None,
        help="Optional JSON/JSONL file containing harmful_prompt values to evaluate.",
    )
    parser.add_argument(
        "--prompt-limit",
        type=int,
        default=0,
        help="Optional max number of prompts to keep from --prompt-file.",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default=None,
        help="Optional output metric filename stem. Use with a single metric.",
    )
    args = parser.parse_args()
    if args.metrics:
        eval_list = args.metrics

prompt_filter = None
if __name__ == "__main__" and args.prompt_file:
    prompt_filter = set()
    with open(args.prompt_file, "r", encoding="utf-8") as f:
        if args.prompt_file.endswith(".jsonl"):
            prompt_rows = [json.loads(line) for line in f if line.strip()]
        else:
            prompt_rows = json.load(f)
    if isinstance(prompt_rows, dict):
        prompt_rows = prompt_rows.get("data", prompt_rows.get("prompts", []))
    for row in prompt_rows[: args.prompt_limit or None]:
        if isinstance(row, str):
            prompt_filter.add(row.strip())
        elif isinstance(row, dict) and row.get("harmful_prompt"):
            prompt_filter.add(row["harmful_prompt"].strip())
    print(f"Loaded {len(prompt_filter)} prompt filters from {args.prompt_file}")

infile_path = "../data/dataset.jsonl"
output_dir = "../temp_results"

print("Loading reference responses...")
reference_map = {}
with open(infile_path, 'r', encoding='utf-8') as f:
    for line in f:
        data = json.loads(line.strip())
        if data.get('response_category') == 'reference response':
            prompt_id = data.get('prompt_id')
            reference_map[prompt_id] = data.get('response', '')
print(f"Loaded {len(reference_map)} reference responses")

for metric_name in eval_list:
    if metric_name not in METRIC_MAPPING:
        print(f"Error: Unknown metric '{metric_name}'. Please check METRIC_MAPPING.")
        continue
    
    blk = METRIC_MAPPING[metric_name]
    
    print(f"\n{'='*60}")
    print(f"Starting evaluation: {metric_name} -> {blk}")
    print(f"{'='*60}\n")

    model = None
    tokenizer = None
    device = None
    
    if blk == "GPTF_eval":
        model, tokenizer, device = curr_metrics.GPTF_init()
    elif blk == "HarmBench_eval":
        model, tokenizer = curr_metrics.HarmBench_init()
    elif blk == "llama_guard_1_eval":
        model, tokenizer = curr_metrics.llama_guard_init("/hub/huggingface/models/meta/LlamaGuard-7b")
    elif blk == "llama_guard_4_eval":
        model, tokenizer = curr_metrics.llama_guard_4_init("/hub/huggingface/models/meta/Llama-Guard-4-12B")
    elif blk == "wildguard_eval":
        model, tokenizer = curr_metrics.wildguard_init("/hub/huggingface/models/allenai/wildguard")
    elif blk == "shieldgemma_eval":
        model, tokenizer = curr_metrics.shieldgemma_init("/hub/huggingface/models/google/shieldgemma-9b")
    elif blk == "Shen_eval":
        model, tokenizer = curr_metrics.Shen_init()
    elif blk == "qwen_harmfulness_classifier":
        model, tokenizer = curr_metrics.qwen_harmfulness_classifier_init()
    elif blk == "qwen35_08b_harmjudge":
        model, tokenizer = curr_metrics.qwen35_08b_harmjudge_init()

    def get_function_by_name(func_name):
        if hasattr(curr_metrics, func_name):
            return getattr(curr_metrics, func_name)
        else:
            raise ValueError(f"Function {func_name} not found in curr_metrics!")

    f_name = blk
    if "rouge" in f_name:
        f_name = "rouge"
    if "llama_guard" in f_name:
        if f_name == "llama_guard_4_eval":
            f_name = "llama_guard_4_eval"
        else:
            f_name = "llama_guard_eval"
        
    f_eval = get_function_by_name(f_name)

    if __name__ == "__main__" and args.output_name:
        if len(eval_list) != 1:
            raise ValueError("--output-name can only be used when evaluating one metric")
        safe_metric_name = args.output_name.replace("/", "-").replace(" ", "_")
    else:
        safe_metric_name = metric_name.replace("/", "-").replace(" ", "_")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    outfile_path = os.path.join(output_dir, f"{safe_metric_name}.jsonl")

    def try_again(func: callable, string0: str, string1: str):
        for i in range(2):
            ret = func(string0, string1)
            if ret < 0:
                print(f"Ret: {ret}. Retrying...")
                time.sleep(1)
                continue
            break
        return ret
    
    processed_ids = set()
    if os.path.exists(outfile_path):
        with open(outfile_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    processed_ids.add(data.get('response_id'))
                except:
                    print("Error reading existing results file!")
                    exit(0)
        print(f"Found {len(processed_ids)} already processed responses, resuming...")
    
    with open(infile_path, 'r', encoding='utf-8') as infile, \
         open(outfile_path, 'a', encoding='utf-8') as outfile:
        
        idx = 0
        fail_num = 0
        skipped_num = 0
        
        for line in infile:
            idx += 1
            
            data = json.loads(line.strip())
            response_id = data.get('response_id')
            
            if response_id in processed_ids:
                continue
            
            harmful_prompt = data.get('harmful_prompt', '')
            response = data.get('response', '')
            response_category = data.get('response_category', '')
            prompt_id = data.get('prompt_id')

            if prompt_filter is not None and harmful_prompt.strip() not in prompt_filter:
                continue
            
            if response_category == "reference response":
                skipped_num += 1
                continue
            
            print(f"{metric_name} - response_id={response_id}")
            
            if not response:
                print(f"Warning: Empty response for response_id={response_id}, skipping...")
                continue
            
            score = None
            
            if blk in ["GCG_eval", "AutoDAN_eval"]:
                score = f_eval(response)
            
            elif "rouge" in blk:
                ref_res = reference_map.get(prompt_id, '')
                if not ref_res:
                    print(f"Warning: No reference response found for prompt_id={prompt_id}, skipping...")
                    continue
                score = f_eval(ref_res, response, blk)
            
            elif blk in ["squad_f1_score", "cos_sim", "meteor", "bleu", "BERT_sim"]:
                ref_res = reference_map.get(prompt_id, '')
                if not ref_res:
                    print(f"Warning: No reference response found for prompt_id={prompt_id}, skipping...")
                    continue
                score = f_eval(ref_res, response)
            
            elif blk == "GPTF_eval":
                score = f_eval(model, tokenizer, device, response)
            
            elif blk == "HarmBench_eval":
                score = f_eval(model, tokenizer, harmful_prompt, response)

            elif blk in ["Qi_eval", 
                        "PAIR_eval", 
                        "JbB_eval",
                        "strongreject_eval",
                        "new_prompt_eval",
                        "Shen_eval",
                        "llama_guard_1_eval",
                        "llama_guard_4_eval",
                        "wildguard_eval",
                        "shieldgemma_eval",
                        "qwen_harmfulness_classifier",
                        "qwen35_08b_harmjudge",
                        ]:
                
                prompt_param = harmful_prompt
                if "llama_guard" in blk or blk == "Shen_eval" or blk == "wildguard_eval" \
                   or blk == "shieldgemma_eval" or blk == "qwen_harmfulness_classifier" \
                   or blk == "qwen35_08b_harmjudge":
                    prompt_param = [model, tokenizer, harmful_prompt]
                
                temp_scores = []
                fail_flag = False
                
                for round_num in range(3):
                    round_score = try_again(f_eval, prompt_param, response)
                    
                    if round_score < 0:
                        fail_flag = True
                        break
                    
                    temp_scores.append(round_score)
                    print(f"\rRound {round_num + 1}/3: score={round_score:.4f}", end="", flush=True)
                
                if fail_flag:
                    print("\nEvaluation failed!")
                    fail_num += 1
                    continue
                
                score = sum(temp_scores) / len(temp_scores)
                print(f"\nAverage score: {score:.4f}")
            
            else:
                print(f"Unknown metric type: {blk}")
                continue
            
            result = {
                'response_id': response_id,
                'score': score
            }
            
            json.dump(result, outfile, ensure_ascii=False)
            outfile.write('\n')
            outfile.flush()
            if args.limit and idx - skipped_num >= args.limit:
                print(f"Reached --limit={args.limit}; stopping {metric_name}.")
                break
        
        print(f'Total failures for {metric_name}: {fail_num}')
        print(f"Results saved to: {outfile_path}")

print("\n" + "="*60)
print("All evaluations completed!")
print("="*60)
