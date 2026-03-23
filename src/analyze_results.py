import json
import matplotlib.pyplot as plt
import numpy as np
import os

def main():
    with open('results/transformers_standard/results.json', 'r') as f:
        standard = json.load(f)
    with open('results/transformers_antithetic/results.json', 'r') as f:
        antithetic = json.load(f)
    
    labels = ['Clean', 'Adversarial', 'OOD']
    base_scores = [standard['base_results']['clean'] * 100, 
                   standard['base_results']['adv'] * 100, 
                   standard['base_results']['ood'] * 100]
    std_scores = [standard['ensemble_results']['clean'] * 100, 
                  standard['ensemble_results']['adv'] * 100, 
                  standard['ensemble_results']['ood'] * 100]
    anti_scores = [antithetic['ensemble_results']['clean'] * 100, 
                   antithetic['ensemble_results']['adv'] * 100, 
                   antithetic['ensemble_results']['ood'] * 100]
    
    x = np.arange(len(labels))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width, base_scores, width, label='Base Model', color='gray')
    rects2 = ax.bar(x, std_scores, width, label='Standard Ensemble', color='skyblue')
    rects3 = ax.bar(x + width, anti_scores, width, label='Antithetic Ensemble', color='salmon')
    
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Robustness of Neural Thicket Ensembles')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    
    ax.bar_label(rects1, padding=3)
    ax.bar_label(rects2, padding=3)
    ax.bar_label(rects3, padding=3)
    
    fig.tight_layout()
    plt.savefig('figures/robustness_comparison.png')
    
    # Also save a summary table
    print("| Metric | Base | Standard | Antithetic |")
    print("|---|---|---|---|")
    for i, label in enumerate(labels):
        print(f"| {label} | {base_scores[i]:.1f}% | {std_scores[i]:.1f}% | {anti_scores[i]:.1f}% |")

if __name__ == "__main__":
    main()
