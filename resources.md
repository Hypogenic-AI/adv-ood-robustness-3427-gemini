# Resources Catalog

### Summary
This document catalogs all resources gathered for the research project, including papers, datasets, and code repositories.

### Papers
Total papers downloaded: 6

| Title | Authors | Year | File | Key Info |
|-------|---------|------|------|----------|
| Neural Thickets | Gan & Isola | 2026 | papers/2603.12228_Neural_Thickets.pdf | RandOpt method |
| Seasoning Model Soups | Croce et al. | 2023 | papers/2302.10164_Seasoning_Model_Soups.pdf | Robustness via soups |
| Model Soups | Wortsman et al. | 2022 | papers/2203.05482_Model_Soups.pdf | Original soup concept |
| RADIN | Menes & Risser-Maroix | 2024 | papers/2401.17790_RADIN.pdf | Faster soups via approximations |
| DARE | de Mathelin et al. | 2023 | papers/2303.02324_DARE.pdf | OOD uncertainty quantification |
| SED | Rubinstein et al. | 2024 | papers/2406.01463_SED.pdf | Scalable ensemble diversification |

### Datasets
Total datasets downloaded: 2

| Name | Source | Size | Task | Location | Notes |
|------|--------|------|------|----------|-------|
| GSM8K | HuggingFace | 7.5K | Math Reasoning | datasets/gsm8k/ | Standard reasoning benchmark |
| MBPP | HuggingFace | 1K | Programming | datasets/mbpp/ | Python code generation |

### Code Repositories
Total repositories cloned: 2

| Name | URL | Purpose | Location | Notes |
|------|-----|---------|----------|-------|
| RandOpt | github.com/sunrainyg/RandOpt | Implementation of RandOpt | code/RandOpt/ | Original repo from Gan & Isola |
| ModelSoup | github.com/milo-sobral/ModelSoup | Implementation of Model Soups | code/ModelSoup/ | Robust reproductions and recipes |

### Recommendations for Experiment Design
- **Primary dataset**: GSM8K for measuring reasoning robustness.
- **Baseline methods**: Standard RandOpt and Single Pretrained Model.
- **Evaluation metrics**: Adversarial accuracy and distribution shift resilience.
- **Proposed Method**: Implementation of **Orthogonal RandOpt** (to maximize expert diversity) and **Adversarial RandOpt** (to find robust expert directions).
