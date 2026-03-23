# Downloaded Datasets

This directory contains datasets for the research project. Data files are NOT committed to git due to size.

## Dataset 1: GSM8K
- **Source**: HuggingFace (openai/gsm8k)
- **Size**: 7473 train, 1319 test
- **Task**: Grade school math word problems.

## Dataset 2: MBPP
- **Source**: HuggingFace (google-research-datasets/mbpp)
- **Size**: 374 train, 500 test, 90 validation
- **Task**: Programming problems for Python code generation.

### Download Instructions
To reload from disk:
```python
from datasets import load_from_disk
ds = load_from_disk('datasets/gsm8k')
```
