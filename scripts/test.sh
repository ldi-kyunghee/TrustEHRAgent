  # OpenAI 모델
  python main.py --llm gpt-3.5-turbo --dataset mimic_iii --num_questions 1

  # Llama 모델 (Ollama 통해)
  python main.py --llm llama-2-7b-chat --dataset mimic_iii --num_questions 1

  # Qwen 모델 (Ollama 통해)  
  python main.py --llm qwen-14b-chat --dataset mimic_iii --num_questions 1

  # Hugging Face 모델 직접
  python main.py --llm mistral-7b --dataset mimic_iii --num_questions 1