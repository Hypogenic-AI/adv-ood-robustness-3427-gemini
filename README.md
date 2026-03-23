# Neural Thicket Robustness

This project investigates the adversarial and out-of-distribution (OOD) robustness of neural thicket ensembles (RandOpt). We specifically compare standard Gaussian sampling with Antithetic sampling to understand the role of expert diversity in weight-space ensembles.

## Key Findings
- **Standard RandOpt** improves clean accuracy but can degrade robustness to OOD distribution shifts.
- **Antithetic RandOpt** (using paired $+\delta$ and $-\delta$ perturbations) significantly improves OOD resilience while maintaining baseline performance.
- Neural thicket ensembles provide a low-compute path to improving model reliability against small adversarial perturbations.

## How to Reproduce
1.  **Setup Environment**:
    ```bash
    uv venv
    source .venv/bin/activate
    uv pip install torch transformers datasets matplotlib tqdm
    ```
2.  **Generate Robust Data**:
    ```bash
    python src/generate_robust_data.py
    ```
3.  **Run Experiments**:
    ```bash
    # Standard Ensemble
    python src/transformers_randopt.py --method standard --output_dir results/transformers_standard
    # Antithetic Ensemble
    python src/transformers_randopt.py --method antithetic --output_dir results/transformers_antithetic
    ```
4.  **Analyze Results**:
    ```bash
    python src/analyze_results.py
    ```

## File Structure
- `src/transformers_randopt.py`: Main implementation of RandOpt using the `transformers` library.
- `src/generate_robust_data.py`: Script to generate adversarial and OOD versions of GSM8K.
- `src/analyze_results.py`: Visualization and summary table generator.
- `REPORT.md`: Detailed research report with results and analysis.

For more details, see [REPORT.md](REPORT.md).
