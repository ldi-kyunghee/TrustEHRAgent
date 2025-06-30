# Multi-Model Support for EhrAgent

EhrAgent now supports multiple model backends through Together AI, OpenAI, Claude, and Gemini APIs.

## Supported Models

### Together AI Models (Default for most models)
- **Llama**: `meta-llama/Llama-2-7b-chat-hf`, `meta-llama/Llama-2-13b-chat-hf`, `meta-llama/Code-Llama-7b-Instruct-hf`
- **Qwen**: `Qwen/Qwen2-72B-Instruct`, `Qwen/Qwen1.5-14B-Chat`, `Qwen/Qwen1.5-7B-Chat`
- **Mistral**: `mistralai/Mistral-7B-Instruct-v0.1`, `mistralai/Mixtral-8x7B-Instruct-v0.1`
- **Other Models**: Any model available on Together AI platform

### OpenAI Models
- `gpt-3.5-turbo`
- `gpt-4`
- `gpt-4.1-mini`
- Any other GPT model

### Other API Models
- **Claude**: `claude-3-sonnet`, `claude-3-opus`, `claude-3-haiku`
- **Gemini**: `gemini-pro`, `gemini-1.5-pro`

## Setup

### 1. Environment Variables

Create a `.env` file or set environment variables:

```bash
# Together AI (default for most models)
TOGETHER_API_KEY=your_together_api_key

# OpenAI (required for GPT models)
OPENAI_API_KEY=your_openai_api_key

# Other APIs (optional)
ANTHROPIC_API_KEY=your_anthropic_key
GOOGLE_API_KEY=your_google_key
AZURE_OPENAI_API_KEY=your_azure_key
AZURE_OPENAI_ENDPOINT=your_azure_endpoint
```

### 2. Install Dependencies

For basic usage:
```bash
pip install openai requests python-dotenv
```

## Usage

### Basic Usage

```bash
# OpenAI GPT models
python main.py --llm gpt-3.5-turbo --dataset mimic_iii --num_questions 1

# Together AI models (Llama)
python main.py --llm meta-llama/Llama-2-7b-chat-hf --dataset mimic_iii --num_questions 1

# Together AI models (Qwen)
python main.py --llm Qwen/Qwen2-72B-Instruct --dataset mimic_iii --num_questions 1

# Together AI models (Mistral)
python main.py --llm mistralai/Mistral-7B-Instruct-v0.1 --dataset mimic_iii --num_questions 1

# Claude models
python main.py --llm claude-3-sonnet --dataset mimic_iii --num_questions 1
```

### Model Backend Selection

The system automatically selects the appropriate backend based on the model name:

- **`gpt-*`** → OpenAI API
- **`claude*`** → Anthropic API
- **`gemini*`** → Google API
- **All others** → Together AI (default)

## Confidence Scoring Support

All models support **embedded confidence scoring** (extracting `# Confidence: X` from responses).

**LogU confidence scoring** (logprobs-based) is currently supported for:
- ✅ OpenAI models (with `logprobs=True`)
- ✅ Together AI models (OpenAI-compatible API with logprobs support)
- ❌ Anthropic Claude models (no logprobs in API)
- ❌ Google Gemini models (no logprobs in API)

## Popular Together AI Models

### Recommended Models by Use Case

**For Chat/General Use:**
- `meta-llama/Llama-2-7b-chat-hf` - Good balance of performance and speed
- `meta-llama/Llama-2-13b-chat-hf` - Better performance, slower
- `mistralai/Mistral-7B-Instruct-v0.1` - Fast and efficient
- `Qwen/Qwen2-72B-Instruct` - High quality, larger model

**For Code Generation:**
- `meta-llama/Code-Llama-7b-Instruct-hf` - Specialized for code
- `WizardLM/WizardCoder-15B-V1.0` - Strong coding capabilities

**For Reasoning Tasks:**
- `NousResearch/Nous-Hermes-2-Mixtral-8x22B-DPO` - Excellent reasoning
- `mistralai/Mixtral-8x7B-Instruct-v0.1` - Good mix of capabilities

## Troubleshooting

### "Client not available" errors
- **OpenAI**: Check API key and internet connection
- **Together AI**: Check API key and model availability
- **Anthropic**: Check API key
- **Google**: Check API key and region availability

### Model not found errors
- Check the exact model name on Together AI platform
- Some models may require approval or payment

### Rate limiting
- Together AI has rate limits based on your plan
- Consider using smaller models for testing

## Getting API Keys

### Together AI
1. Sign up at https://api.together.xyz/
2. Get API key from dashboard
3. Choose your pricing plan

### OpenAI
1. Sign up at https://platform.openai.com/
2. Add billing information
3. Generate API key

### Anthropic Claude
1. Sign up at https://console.anthropic.com/
2. Request access to Claude API
3. Generate API key

### Google Gemini
1. Sign up at https://makersuite.google.com/
2. Enable Gemini API
3. Generate API key