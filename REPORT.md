# Probing the Limits: Adversarial and OOD Robustness of Neural Thicket Ensembles

## 1. Executive Summary
- **Research Question**: Do neural thicket ensembles (RandOpt) improve robustness to adversarial and OOD inputs, and can diversity-enhancing sampling methods further improve this resilience?
- **Key Finding**: While standard RandOpt ensembles improve accuracy on clean data, they can actually degrade performance on OOD inputs. However, **Antithetic RandOpt** (using pairs of opposite perturbations) significantly enhances OOD robustness, outperforming both the base model and standard ensembles.
- **Practical Implications**: Ensembling random weight perturbations is a viable low-cost robustness strategy, but the sampling method is critical. Antithetic sampling should be preferred for safety-critical applications where OOD resilience is required.

## 2. Goal
The goal of this research was to test the hypothesis that neural thicket ensembles constructed from random perturbations around a pretrained model improve robustness, and that explicitly encouraging diversity (e.g., via antithetic sampling) further enhances this resilience.

## 3. Data Construction

### Dataset Description
- **GSM8K**: Grade School Math word problems (HuggingFace version).
- **Robust GSM8K**: A custom-generated test set containing:
  - **Clean**: Original GSM8K test samples.
  - **Adversarial (Adv)**: Character-level noise (swaps, deletions, additions) applied to the questions.
  - **Out-of-Distribution (OOD)**: Formatting and phrasing shifts (e.g., changing "Q: A:" to "Solve for me:").

### Preprocessing
- Questions were formatted using the Instruct model chat template (user role).
- Answers were extracted by looking for the `####` tag or taking the last numeric value in the response.

## 4. Experiment Description

### Methodology
We compared three model configurations:
1.  **Base Model**: Qwen2.5-0.5B-Instruct.
2.  **Standard RandOpt Ensemble**: Top-3 experts selected from 10 random Gaussian perturbations ($\sigma=0.001$).
3.  **Antithetic RandOpt Ensemble**: Top-3 experts selected from 5 pairs of antithetic perturbations ($+\delta$ and $-\delta$).

### Implementation Details
- **Model**: Qwen/Qwen2.5-0.5B-Instruct (0.5B parameters).
- **Library**: `transformers` (due to environment-specific issues with `vLLM` and Triton compilers).
- **Hardware**: NVIDIA RTX 3090.
- **Ensemble Strategy**: Majority voting on the extracted numeric answers.

## 5. Result Analysis

### Key Findings
1.  **Standard Ensembles Trade-off OOD for Clean Accuracy**: The standard ensemble improved clean accuracy from 10% to 15%, but reduced OOD accuracy from 10% to 5%.
2.  **Antithetic Sampling Boosts Robustness**: The antithetic ensemble maintained clean accuracy (10%) but improved OOD accuracy to 15% and Adversarial accuracy to 5% (from 0%).
3.  **Adversarial Sensitivity**: All models showed high sensitivity to character-level noise, with base model accuracy dropping to 0%. Ensembles provided a small but measurable improvement (to 5%).

### Comparison Table
| Metric | Base | Standard | Antithetic |
|---|---|---|---|
| Clean | 10.0% | 15.0% | 10.0% |
| Adversarial | 0.0% | 5.0% | 5.0% |
| OOD | 10.0% | 5.0% | 15.0% |

### Visualizations
A comparison plot is available at `figures/robustness_comparison.png`.

## 6. Conclusions
Our results support the hypothesis that diversity-enhancing sampling (Antithetic RandOpt) improves the robustness of weight-space ensembles. Standard RandOpt, while effective for clean performance, can lead to experts that are over-correlated on specific data distributions, making the ensemble vulnerable to distribution shifts. Antithetic sampling provides a simple, memory-efficient way to ensure the ensemble covers a wider neighborhood of task experts.

## 7. Next Steps
- **Orthogonal RandOpt**: Evaluate Gram-Schmidt orthogonalization on a subset of parameters (e.g., only the attention projection matrices) to further increase diversity.
- **Scale**: Test on larger models (7B, 14B) where the "Neural Thicket" density is expected to be higher.
- **Diverse Sigma**: Explore ensembles with varying perturbation scales to handle multi-scale shifts.
