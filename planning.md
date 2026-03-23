# Research Plan: Probing the Limits of Neural Thicket Ensembles

## Motivation & Novelty Assessment

### Why This Research Matters
Neural thicket ensembles (RandOpt) offer a computationally efficient way to create task-specific experts from pretrained models. However, their security and robustness properties are not yet understood. As LLMs are increasingly deployed in real-world scenarios, understanding how these ensembles handle adversarial attacks and out-of-distribution (OOD) data is critical.

### Gap in Existing Work
The "Neural Thickets" hypothesis (Gan & Isola, 2026) demonstrates performance gains on clean benchmarks, but does not explore robustness. While "Model Soups" has some literature on robustness (Croce et al., 2023), the specific case of ensembles formed by random perturbations remains a mystery. There is no existing work on whether random directions provide inherent robustness or if they are equally vulnerable to targeted attacks.

### Our Novel Contribution
We propose and evaluate **Orthogonal RandOpt**, which explicitly enforces diversity in the weight space to improve ensemble resilience. We also provide the first systematic evaluation of RandOpt's robustness to adversarial and OOD shifts on reasoning tasks (GSM8K).

### Experiment Justification
- **Experiment 1 (Baseline Robustness)**: Establish the robustness of the original RandOpt method to provide a benchmark.
- **Experiment 2 (Orthogonal RandOpt Implementation & Evaluation)**: Determine if weight-space diversity translates to adversarial robustness.
- **Experiment 3 (Ablation on Ensemble Size)**: Study how the number of experts and the sampling ratio affect robustness.

---

## Research Question
Do neural thicket ensembles improve adversarial and OOD robustness compared to single models, and can explicit diversity (orthogonality) further enhance this resilience?

## Hypothesis Decomposition
- **H1**: Weight-space ensembles (RandOpt) are more robust than single models to stochastic perturbations in the input.
- **H2**: Standard RandOpt experts may be highly correlated, making them vulnerable to the same adversarial directions.
- **H3**: Orthogonalizing the perturbations in weight space increases the diversity of the experts' failure modes, leading to higher ensemble robustness.

## Proposed Methodology

### Approach
1.  **Environment Setup**: Utilize `uv` for dependency management and `vllm`/`ray` for model serving.
2.  **Dataset Preparation**: Use GSM8K. Generate adversarial examples using character-level perturbations and word substitutions.
3.  **Baseline Execution**: Run the original RandOpt implementation on Qwen2.5-3B-Instruct.
4.  **Orthogonal RandOpt Implementation**: Modify the sampling procedure in `randopt.py` to ensure perturbations are orthogonal.
5.  **Comparative Analysis**: Compare Clean Accuracy, Adversarial Accuracy, and OOD Accuracy across all methods.

### Baselines
- Single Pretrained Model.
- Standard RandOpt (Gaussian).
- Logit-Space Ensemble (Averaging outputs of the same RandOpt experts).

### Evaluation Metrics
- Accuracy (Clean, Adversarial, OOD).
- Disagreement Score: Fraction of samples where experts disagree.
- Perturbation Similarity: Cosine similarity between weight perturbations.

## Success Criteria
- Successful implementation of Orthogonal RandOpt.
- Statistical significance (p < 0.05) in robustness improvements over the single model baseline.
- Clear characterization of RandOpt's vulnerability/robustness to adversarial attacks.
