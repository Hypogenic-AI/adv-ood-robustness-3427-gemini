import random
import os
import json
from datasets import load_from_disk
from tqdm import tqdm

def apply_character_noise(text, p=0.05):
    """Randomly swap, delete, or add characters with probability p."""
    chars = list(text)
    for i in range(len(chars)):
        if random.random() < p:
            mode = random.choice(['swap', 'delete', 'add'])
            if mode == 'swap' and i < len(chars) - 1:
                chars[i], chars[i+1] = chars[i+1], chars[i]
            elif mode == 'delete' and len(chars) > 1:
                chars[i] = ''
            elif mode == 'add':
                chars[i] += random.choice('abcdefghijklmnopqrstuvwxyz')
    return ''.join(chars)

def apply_formatting_shift(question):
    """Change the problem description to be less formal."""
    shifts = [
        lambda q: f"Hey, can you solve this? {q}",
        lambda q: f"Solve for me please: {q}",
        lambda q: f"MATH PROBLEM: {q}",
        lambda q: f"Help Natalia and friends: {q}" if "Natalia" in q else f"Problem for you: {q}",
        lambda q: q.replace("Natalia", "Nancy").replace("April", "January") # Simple name/date shift
    ]
    return random.choice(shifts)(question)

def main():
    random.seed(42)
    os.makedirs('datasets/gsm8k_robust', exist_ok=True)
    
    ds = load_from_disk('datasets/gsm8k')
    test_ds = ds['test']
    
    robust_data = []
    
    print("Generating robust GSM8K test set...")
    for i, item in enumerate(tqdm(test_ds)):
        q = item['question']
        a = item['answer']
        
        # 1. Clean (Original)
        # 2. Adversarial (Character Noise)
        q_adv = apply_character_noise(q, p=0.02)
        
        # 3. OOD (Formatting Shift)
        q_ood = apply_formatting_shift(q)
        
        robust_data.append({
            'index': i,
            'question_clean': q,
            'question_adv': q_adv,
            'question_ood': q_ood,
            'answer': a,
            'ground_truth': a.split('####')[-1].strip()
        })
    
    with open('datasets/gsm8k_robust/test_robust.json', 'w') as f:
        json.dump(robust_data, f, indent=4)
    
    print(f"Saved {len(robust_data)} samples to datasets/gsm8k_robust/test_robust.json")

if __name__ == "__main__":
    main()
