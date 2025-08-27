# 🛡️ TrustEHRAgent ⚕️

[![arXiv](https://img.shields.io/badge/arXiv-TBA-b31b1b.svg)](https://arxiv.org/abs/2508.19096)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)  


The official repository for the code of the paper "[Trustworthy Agents for Electronic Health Records through Confidence Estimation](https://arxiv.org/abs/2508.19096)".
TrustEHRAgent is a confidence-aware LLM agent that incorporates step-wise confidence estimation for reliable clinical question answering on Electronic Health Records (EHRs). This work addresses critical hallucination risks in clinical AI systems by introducing novel confidence estimation mechanisms.

## 🌟 Features

- **Confidence-Aware Clinical AI**: TrustEHRAgent incorporates step-wise confidence estimation to address hallucination risks in clinical decision support
- **HCAcc@k% Metric**: Novel evaluation metric quantifying the accuracy-reliability trade-off at varying confidence thresholds
- **Trustworthy Clinical Decision Support**: Delivers accurate information when confident, or transparently expresses uncertainty when confidence is low
- **Strong Reliability Performance**: Achieves substantial improvements of 44.23%p and 25.34%p at HCAcc@70% on MIMIC-III and eICU datasets

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- [uv](https://docs.astral.sh/uv/) package manager
- OpenAI API key

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/TrustEHRAgent.git
   cd TrustEHRAgent
   ```

2. **Set up environment**
   ```bash
   uv sync
   source .venv/bin/activate
   ```

3. **Configure credentials**
   
   Add your OpenAI API key to `./ehragent/config.py`:
   ```python
   OPENAI_API_KEY = "your-api-key-here"
   ```

### Data Setup

Prepare MIMIC-III and eICU datasets following the [EHRAgent data preparation guide](https://github.com/wshi83/EHRAgent#data-preparation).

## 📋 Usage

### Basic Execution

**MIMIC-III Dataset:**
```bash
./scripts/run_code.sh
```

**eICU Dataset:**
```bash
./scripts/run_code_eicu.sh
```

### Confidence Analysis

**Run confidence estimation and get HCAcc@k% metrics:**
```bash
./scripts/run_evaluate.sh
```

### Results

All results are automatically saved to `./logs/` directory with detailed confidence scores and performance metrics.

## 📚 Citation

If you find this work useful, please cite:

```bibtex
@article{trustehragent2024,
    title = "Trustworthy Agents for Electronic Health Records through Confidence Estimation",
    author = "TBD",
    journal = "TBD",
    year = "2025",
    abstract = "Large language models (LLMs) have emerged as a promising approach for extracting and interpreting information from Electronic Health Records (EHR), offering possibilities for clinical decision support and patient care. However, despite their capabilities, the reliable deployment of LLMs in clinical settings remains challenging due to risks of hallucinations. Thus, we propose Hallucination-Controlled Accuracy at k% (HCAcc@k%), a novel metric quantifying the accuracy-reliability trade-off at varying confidence thresholds. Based on this framework, we also introduce TrustEHRAgent, a confidence-aware agent that incorporates step-wise confidence estimation for clinical question answering. Experiments on MIMIC-III and eICU datasets demonstrate that TrustEHRAgent outperforms baselines under strict reliability constraints, achieving substantial improvements of 44.23%p and 25.34%p at HCAcc@70% while baseline methods fail completely at these stringent thresholds."
}
```

## 🙏 Acknowledgments

This work builds upon [EHRAgent](https://github.com/wshi83/EHRAgent). We thank the original authors for their foundational contributions to code-empowered agent systems.


<div align="center">
<strong>⚠️ Important:</strong> This tool is for research purposes only. Always consult healthcare professionals for medical decisions.
</div>
