# Literature Review: Adversarial and OOD Robustness of Neural Thicket Ensembles

## Research Area Overview
This research investigates the robustness of ensembles created through random perturbations around pretrained model weights, a concept recently popularized by the "Neural Thickets" hypothesis. We focus on how the diversity and independence of these perturbations affect their resilience to adversarial attacks and out-of-distribution (OOD) shifts.

## Key Papers

### 1. Neural Thickets: Diverse Task Experts Are Dense Around Pretrained Weights (Gan & Isola, 2026)
- **Key Contribution**: Proposes the "Neural Thicket" hypothesis, stating that large pretrained models are surrounded by a dense neighborhood of task-specific experts.
- **Methodology**: Introduces **RandOpt**, a method that samples N random Gaussian perturbations around pretrained weights and ensembles the top K performers.
- **Results**: Shows that RandOpt is competitive with standard post-training methods (PPO, GRPO) and that the sampled experts are diverse specialists.
- **Relevance**: Foundational for our study; defines the ensemble method we are testing.

### 2. Seasoning Model Soups for Robustness to Adversarial and Natural Distribution Shifts (Croce et al., 2023)
- **Key Contribution**: Demonstrates that "soups" (weight averaging) of models fine-tuned with different adversarial objectives can improve robustness to multiple threats.
- **Methodology**: Fine-tunes models for different lp norms and interpolates their weights.
- **Results**: Soups can smoothly trade off between different types of robustness and often outperform individual models.
- **Relevance**: Provides a baseline for adversarial robustness in weight-space ensembles.

### 3. Model Soups: Averaging Weights of Multiple Fine-Tuned Models Improves Accuracy without Increasing Inference Time (Wortsman et al., 2022)
- **Key Contribution**: Introduces "Model Soups" - averaging weights of models fine-tuned with different hyperparameters.
- **Results**: Significantly improves performance on ImageNet and other benchmarks without additional compute at inference.
- **Relevance**: The original concept upon which Neural Thickets and Seasoning are built.

## Common Methodologies
- **Weight Averaging / Interpolation**: Combining multiple models in the weight space rather than the logit space.
- **Random Perturbations**: Sampling around a central point (pretrained weights) to find high-performing directions.
- **Diversity Encouragement**: Traditionally done via hyperparameter variation; we propose orthogonal or adversarial directions.

## Standard Baselines
- **Single Pretrained Model**: The starting point for perturbations.
- **Standard RandOpt**: Gaussian random perturbations (from Gan & Isola, 2026).
- **Logit-Space Ensembles**: Majority voting or averaging of model outputs.

## Evaluation Metrics
- **Clean Accuracy**: Performance on standard benchmarks (GSM8K, MBPP).
- **Adversarial Accuracy**: Performance under attacks (e.g., character-level perturbations, word substitutions for LLMs).
- **OOD Accuracy**: Performance on distribution-shifted versions of the task.

## Recommendations for Our Experiment
- **Datasets**: GSM8K and MBPP (as used in Neural Thickets), plus adversarial versions (e.g., GSM8K-Robustness).
- **Baselines**: Standard RandOpt, Uniform Soup of fine-tuned models.
- **Proposed Method**: Orthogonal RandOpt (Gram-Schmidt on perturbations) and Adversarial RandOpt (using gradients to find "diverse" directions).
