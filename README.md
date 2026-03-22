# LLM-Bias-Miner 🔍

**Systematic Bias Discovery in Large Language Models Using Data Mining Methods**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Hugging Face](https://img.shields.io/badge/🤗-Hugging%20Face-yellow.svg)](https://huggingface.co/)

## Overview

LLM-Bias-Miner applies classical data mining techniques — **association rule mining, subgroup discovery, anomaly detection, and statistical pattern analysis** — to systematically audit Large Language Models for bias and fairness violations.

Unlike existing fairness toolkits that focus on traditional ML classifiers, this project treats **LLM bias auditing as a pattern discovery problem**: given a large corpus of LLM responses across demographic variations, what hidden bias patterns can we mine?

**Tech Stack:** PyTorch · Hugging Face Transformers · Sentence-Transformers · Hugging Face Evaluate

### Key Contributions

1. **Association Rule Mining for Bias** — Discovers co-occurrence patterns between demographic attributes and sentiment/toxicity in LLM outputs (e.g., `{gender=female, topic=leadership} → {sentiment=negative}`)
2. **Subgroup Discovery for Fairness** — Identifies subpopulations where the LLM exhibits disproportionate bias using beam search over attribute combinations
3. **Anomaly-Based Bias Detection** — Uses PyTorch autoencoders and embedding analysis to flag demographic groups receiving anomalously different treatment
4. **Statistical Bias Testing** — Systematic hypothesis testing framework with multiple comparison correction across all demographic intersections

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│                   Prompt Generator                        │
│   Demographic templates × Scenarios × Occupations         │
└────────────────────────┬─────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────┐
│             LLM Collector (HF Transformers)               │
│   AutoModelForCausalLM / pipeline / API-based inference   │
└────────────────────────┬─────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────┐
│         Response Analyzer (HF Pipelines + PyTorch)        │
│   Sentiment · Toxicity · Regard · Embeddings · Refusal    │
└────────────────────────┬─────────────────────────────────┘
                         │
          ┌──────────────┼──────────────┬───────────────┐
          ▼              ▼              ▼               ▼
   ┌─────────────┐┌───────────┐┌────────────┐┌──────────────┐
   │ Association  ││  Subgroup  ││  Anomaly   ││  Statistical  │
   │    Rule      ││ Discovery  ││ Detection  ││   Testing     │
   │   Mining     ││            ││ (PyTorch)  ││              │
   └─────────────┘└───────────┘└────────────┘└──────────────┘
          │              │              │               │
          └──────────────┴──────────────┴───────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────┐
│            Bias Report & Model Card Generator              │
│   Visualizations · Findings · Policy Recommendations       │
└──────────────────────────────────────────────────────────┘
```

## Quick Start

### Installation

```bash
git clone https://github.com/shuwenw168-bot/llm-bias-miner.git
cd llm-bias-miner
pip install -r requirements.txt
```

### Run the Full Audit Pipeline

```bash
# Option 1: Run with sample data (no GPU / API key needed)
python experiments/run_full_audit.py --use-sample-data

# Option 2: Run with a Hugging Face model (GPU recommended)
python experiments/run_full_audit.py --model "meta-llama/Llama-3.1-8B-Instruct"

# Option 3: Run with OpenAI API
export OPENAI_API_KEY="your-key-here"
python experiments/run_full_audit.py --model "openai:gpt-4o-mini"
```

Outputs are saved to `results/` with visualizations in `results/figures/`.

### Jupyter Notebooks

| Notebook | Description |
|----------|-------------|
| `notebooks/01_data_exploration.ipynb` | Explore prompt templates and response distributions |
| `notebooks/02_bias_mining.ipynb` | Step-by-step walkthrough of all four mining methods |
| `notebooks/03_results_visualization.ipynb` | Generate publication-ready figures and model cards |

## Mining Methods in Detail

### 1. Association Rule Mining (`AssociationBiasMiner`)

Adapts the Apriori algorithm to discover frequent co-occurrence patterns between demographic prompt attributes and response characteristics.

```python
from src.miners.association_miner import AssociationBiasMiner

miner = AssociationBiasMiner(min_support=0.05, min_confidence=0.6, min_lift=1.5)
rules = miner.mine(response_df)
bias_rules = miner.filter_bias_rules(rules)

# Example output:
# {gender=female, topic=STEM} → {sentiment=condescending} (lift=2.31, p=0.003)
# {race=Black, scenario=hiring} → {recommendation=negative} (lift=1.87, p=0.011)
```

### 2. Subgroup Discovery (`SubgroupBiasMiner`)

Uses beam search to find demographic subgroups where the LLM's behavior deviates most from the overall population.

```python
from src.miners.subgroup_miner import SubgroupBiasMiner

miner = SubgroupBiasMiner(beam_width=10, max_depth=3, min_subgroup_size=30)
subgroups = miner.discover(response_df, target="toxicity_score")

# Example output:
# Subgroup: {age=elderly, gender=female, topic=technology}
#   Mean toxicity: 0.42 vs population mean: 0.18 (effect_size=1.34)
```

### 3. Anomaly-Based Bias Detection (`AnomalyBiasMiner`)

Uses a PyTorch autoencoder on response embeddings to detect demographic groups that receive anomalously different responses.

```python
from src.miners.anomaly_miner import AnomalyBiasMiner

miner = AnomalyBiasMiner(embedding_model="sentence-transformers/all-MiniLM-L6-v2")
anomalies = miner.detect(response_df)
```

### 4. Statistical Bias Testing (`StatisticalBiasTester`)

Applies systematic hypothesis testing across all demographic intersections with Bonferroni / BH correction.

```python
from src.miners.statistical_tester import StatisticalBiasTester

tester = StatisticalBiasTester(correction_method="benjamini-hochberg", alpha=0.05)
results = tester.test_all(response_df)
significant = tester.get_significant_findings(results)
```

## Project Structure

```
llm-bias-miner/
├── config/
│   └── default_config.yaml          # All hyperparameters and settings
├── src/
│   ├── collectors/
│   │   ├── prompt_generator.py       # Demographic prompt template engine
│   │   ├── llm_querier.py           # HF Transformers / API model querier
│   │   └── sample_generator.py      # Synthetic data with embedded biases
│   ├── miners/
│   │   ├── association_miner.py      # Apriori-based bias rule mining
│   │   ├── subgroup_miner.py         # Beam search subgroup discovery
│   │   ├── anomaly_miner.py          # PyTorch autoencoder anomaly detection
│   │   └── statistical_tester.py     # Hypothesis testing framework
│   ├── metrics/
│   │   ├── response_analyzer.py      # HF pipelines: sentiment/toxicity/regard
│   │   └── fairness_metrics.py       # Demographic parity, equalized odds, etc.
│   ├── visualization/
│   │   └── bias_plots.py            # Publication-ready visualizations
│   └── reporting/
│       └── model_card.py            # Auto-generated bias report
├── experiments/
│   ├── run_full_audit.py            # Main pipeline script
│   └── validate.py                  # Quick validation without all deps
├── data/prompts/
│   └── templates.yaml               # Prompt templates with demographic slots
├── notebooks/                        # Interactive walkthroughs
├── tests/
│   └── test_miners.py               # Unit tests (pytest)
└── results/                          # Generated outputs
```

## Extending This Project

**Add a new demographic dimension:** Edit `data/prompts/templates.yaml`.

**Add a new mining method:** Implement the pattern in `src/miners/` and add to the pipeline in `run_full_audit.py`.

**Audit a new model:**
```bash
python experiments/run_full_audit.py --model "your-org/your-model"
```

Any Hugging Face model with a text-generation pipeline is supported.

## Citation

```bibtex
@software{llm_bias_miner_2026,
  title={LLM-Bias-Miner: Systematic Bias Discovery in Large Language Models Using Data Mining Methods},
  author={[Shuwen Wang]},
  year={2026},
  url={https://github.com/shuwenw168-bot/llm-bias-miner}
}
```

## License

MIT License — see [LICENSE](LICENSE) for details.
