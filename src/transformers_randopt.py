import argparse
import json
import os
import random
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_from_disk
from collections import Counter

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def apply_perturbation(model, seed, sigma, negate=False):
    """Apply Gaussian perturbation to model weights."""
    state_dict = model.state_dict()
    sign = -1.0 if negate else 1.0
    for name, param in model.named_parameters():
        if "visual" in name: continue # Skip visual encoder if any
        
        # Use a generator for reproducibility per parameter
        gen = torch.Generator(device=param.device)
        # We need a unique seed per parameter to avoid correlated noise across layers
        # but deterministic based on the global seed
        param_seed = seed + hash(name) % (2**31)
        gen.manual_seed(param_seed % (2**31))
        
        noise = torch.randn(param.shape, device=param.device, dtype=param.dtype, generator=gen)
        param.data.add_(sign * sigma * noise)

def restore_weights(model, base_state_dict):
    """Restore model weights from a base state dict."""
    model.load_state_dict(base_state_dict)

def extract_answer(text):
    """Extract numeric answer from GSM8K response."""
    # Look for '####'
    if "####" in text:
        return text.split("####")[-1].strip()
    # Simple heuristic if #### is missing
    import re
    nums = re.findall(r"[-+]?\d*\.\d+|\d+", text)
    if nums:
        return nums[-1]
    return ""

def evaluate(model, tokenizer, dataset, device, max_new_tokens=256):
    """Evaluate model on a dataset."""
    model.eval()
    correct = 0
    results = []
    
    for item in tqdm(dataset, desc="Evaluating"):
        prompt = f"Question: {item['question']}\nAnswer: "
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        with torch.no_grad():
            output_ids = model.generate(
                **inputs, 
                max_new_tokens=max_new_tokens, 
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(output_ids[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        pred = extract_answer(response)
        gt = item['ground_truth']
        
        is_correct = (pred == gt)
        if is_correct:
            correct += 1
        
        results.append({
            "response": response,
            "pred": pred,
            "gt": gt,
            "correct": is_correct
        })
    
    accuracy = correct / len(dataset) if len(dataset) > 0 else 0
    return accuracy, results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--population_size", type=int, default=10)
    parser.add_argument("--train_samples", type=int, default=20)
    parser.add_argument("--test_samples", type=int, default=20)
    parser.add_argument("--sigma", type=float, default=0.001)
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument("--method", type=str, choices=["standard", "antithetic"], default="standard")
    parser.add_argument("--output_dir", type=str, default="results/transformers_randopt")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Loading model: {args.model_name}")
    model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.bfloat16).to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    # Save base weights
    base_state_dict = {k: v.clone() for k, v in model.state_dict().items()}
    
    print("Loading datasets...")
    ds = load_from_disk('datasets/gsm8k')
    train_data = []
    for i in range(args.train_samples):
        item = ds['train'][i]
        train_data.append({
            "question": item['question'],
            "ground_truth": item['answer'].split("####")[-1].strip()
        })
    
    # Load robust test data
    with open('datasets/gsm8k_robust/test_robust.json', 'r') as f:
        robust_raw = json.load(f)[:args.test_samples]
    
    test_variants = {
        "clean": [{"question": d['question_clean'], "ground_truth": d['ground_truth']} for d in robust_raw],
        "adv": [{"question": d['question_adv'], "ground_truth": d['ground_truth']} for d in robust_raw],
        "ood": [{"question": d['question_ood'], "ground_truth": d['ground_truth']} for d in robust_raw]
    }

    # 1. Base Model Evaluation
    print("\n--- Base Model Evaluation ---")
    base_results = {}
    for var_name, var_data in test_variants.items():
        acc, _ = evaluate(model, tokenizer, var_data, device)
        base_results[var_name] = acc
        print(f"Base {var_name} Accuracy: {acc*100:.2f}%")

    # 2. Perturbation Sampling
    print(f"\n--- Perturbation Sampling (Method: {args.method}) ---")
    perf = []
    
    seeds = []
    if args.method == "antithetic":
        for i in range(args.population_size // 2):
            s = random.randint(0, 2**31 - 1)
            seeds.append((s, False))
            seeds.append((s, True))
    else:
        for i in range(args.population_size):
            s = random.randint(0, 2**31 - 1)
            seeds.append((s, False))

    for seed, neg in seeds:
        print(f"Sampling seed={seed}, neg={neg}...")
        apply_perturbation(model, seed, args.sigma, negate=neg)
        acc, _ = evaluate(model, tokenizer, train_data, device)
        perf.append({
            "seed": seed,
            "negate": neg,
            "train_acc": acc
        })
        restore_weights(model, base_state_dict)
        print(f"  Train Accuracy: {acc*100:.2f}%")

    # 3. Selection
    perf.sort(key=lambda x: x['train_acc'], reverse=True)
    top_k_experts = perf[:args.top_k]
    print(f"\nSelected top-{args.top_k} experts:")
    for i, exp in enumerate(top_k_experts):
        print(f"  {i+1}. seed={exp['seed']}, neg={exp['negate']}, acc={exp['train_acc']*100:.2f}%")

    # 4. Ensemble Evaluation
    print("\n--- Ensemble Evaluation ---")
    all_expert_answers = {var: [] for var in test_variants} # all_expert_answers[var][expert_idx][sample_idx]
    
    for exp in top_k_experts:
        print(f"Evaluating expert seed={exp['seed']}, neg={exp['negate']}...")
        apply_perturbation(model, exp['seed'], args.sigma, negate=exp['negate'])
        
        for var_name, var_data in test_variants.items():
            _, results = evaluate(model, tokenizer, var_data, device)
            all_expert_answers[var_name].append([r['pred'] for r in results])
        
        restore_weights(model, base_state_dict)

    # Majority Voting
    ensemble_results = {}
    for var_name, var_data in test_variants.items():
        correct = 0
        expert_answers = all_expert_answers[var_name] # [expert_idx][sample_idx]
        num_samples = len(var_data)
        
        for i in range(num_samples):
            answers = [expert_answers[j][i] for j in range(len(top_k_experts)) if expert_answers[j][i]]
            if answers:
                voted_answer = Counter(answers).most_common(1)[0][0]
                if voted_answer == var_data[i]['ground_truth']:
                    correct += 1
        
        ens_acc = correct / num_samples if num_samples > 0 else 0
        ensemble_results[var_name] = ens_acc
        print(f"Ensemble {var_name} Accuracy: {ens_acc*100:.2f}% (Gain: {(ens_acc - base_results[var_name])*100:.2f}%)")

    # Save results
    final_results = {
        "args": vars(args),
        "base_results": base_results,
        "top_k_experts": top_k_experts,
        "ensemble_results": ensemble_results
    }
    with open(os.path.join(args.output_dir, "results.json"), "w") as f:
        json.dump(final_results, f, indent=4)

if __name__ == "__main__":
    main()
