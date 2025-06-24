<div align="center">
<h1> 🛡️TrustEHRAgent⚕️ </h1>
</div>

The official repository for the code of the paper "**Trustworthy Agents for Electronic Health Records through Confidence Estimation**". TrustEHRAgent is a confidence-aware LLM agent that incorporates step-wise confidence estimation for reliable clinical question answering on Electronic Health Records (EHRs).

This work builds upon [EHRAgent](https://github.com/wshi83/EHRAgent), introducing novel confidence estimation mechanisms and the HCAcc@k% metric to address hallucination risks in clinical AI systems.

### Features

- **Confidence-Aware Clinical AI**: TrustEHRAgent incorporates step-wise confidence estimation to address hallucination risks in clinical decision support;
- **HCAcc@k% Metric**: Novel evaluation metric quantifying the accuracy-reliability trade-off at varying confidence thresholds;
- **Trustworthy Clinical Decision Support**: Delivers accurate information when confident, or transparently expresses uncertainty when confidence is low;
- **Strong Reliability Performance**: Achieves substantial improvements of 44.23%p and 25.34%p at HCAcc@70% on MIMIC-III and eICU datasets;
- **Built on EHRAgent**: Extends the code-empowered LLM agent framework with confidence estimation capabilities.

### Data Preparation

TBD

### Credentials Preparation
Our experiments are based on OpenAI API services. Please record your API keys and other credentials in the ``./trustehragent/config.py`` or ``./ehragent/config.py``. 

### Setup

See ``requirements.txt``. Packages with versions specified in ``requirements.txt`` are used to test the code. Other versions that are not fully tested may also work. We also kindly suggest the users to run this code with Python version: ``python>=3.9``. Install required libraries with the following command:

```bash
pip3 install -r requirements.txt
```

### Instructions

The outputting results will be saved under the directory ``./logs/``. We provide convenient execution scripts in the ``scripts/`` directory:

#### Quick Start Scripts

**MIMIC-III Dataset:**
```bash
./scripts/run_code.sh
```

**eICU Dataset:**
```bash
./scripts/run_code_eicu.sh
```

**Confidence Analysis & Evaluation:**
```bash
./scripts/run_evaluate.sh
```

**Test Configuration:**
```bash
./scripts/run_test.sh
```

**Post-Evaluation Analysis:**
```bash
./scripts/run_post_analysis.sh
```

#### Key Features

- **Confidence Estimation**: Each script includes confidence analysis capabilities
- **HCAcc@k% Evaluation**: Reliability-accuracy trade-off evaluation at different confidence thresholds  
- **Dual Dataset Support**: Both MIMIC-III and eICU datasets are supported
- **Comprehensive Analysis**: From execution to post-evaluation visualization

### Citation
If you find this repository useful, please consider citing:
```bibtex
@article{trustehragent2024,
    title = "Trustworthy Agents for Electronic Health Records through Confidence Estimation",
    author = "TBD",
    journal = "TBD",
    year = "2024",
    abstract = "Large language models (LLMs) have emerged as a promising approach for extracting and interpreting information from Electronic Health Records (EHR), offering possibilities for clinical decision support and patient care. However, despite their capabilities, the reliable deployment of LLMs in clinical settings remains challenging due to risks of hallucinations. Thus, we propose Hallucination-Controlled Accuracy at k% (HCAcc@k%), a novel metric quantifying the accuracy-reliability trade-off at varying confidence thresholds. Based on this framework, we also introduce TrustEHRAgent, a confidence-aware agent that incorporates step-wise confidence estimation for clinical question answering. Experiments on MIMIC-III and eICU datasets demonstrate that TrustEHRAgent outperforms baselines under strict reliability constraints, achieving substantial improvements of 44.23%p and 25.34%p at HCAcc@70% while baseline methods fail completely at these stringent thresholds."
}
```

### Acknowledgments
This work builds upon [EHRAgent](https://github.com/wshi83/EHRAgent). We thank the original authors for their foundational contributions to code-empowered clinical decision support systems.
