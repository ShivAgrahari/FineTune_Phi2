Fine-Tuning for Phi-2 Model
Purpose of Fine-Tuning
The Phi-2 model, a Transformer-based model with 2.7 billion parameters, was fine-tuned to enhance its performance in generating coherent, contextually relevant, and concise responses, particularly for creative tasks such as storytelling. The fine-tuning process aimed to improve the model's ability to produce high-quality text in response to prompts like "Tell me a story about a brave warrior." The goal was to address limitations in the base model's output, which often exhibited verbosity, irrelevant text, and textbook-like responses due to its primary training on textbook data. By fine-tuning with the Alpaca dataset, the model was adapted to better handle creative and conversational prompts, improving its ability to generate engaging and focused narratives.
Fine-Tuning Process
The fine-tuning was performed using the following setup:
Dataset

Alpaca Dataset: A dataset containing instruction-output pairs designed for conversational tasks was used. A subset of 5,000 samples was selected for training, and 500 samples for evaluation to optimize for speed and resource constraints.
Tokenization: The dataset was tokenized using the Phi-2 tokenizer, with inputs formatted as instruction followed by output, truncated and padded to a maximum length of 128 tokens. Labels were set to match input IDs for loss computation.

Model Configuration

Base Model: Microsoft Phi-2, loaded with 4-bit quantization (torch.float16) to reduce VRAM usage.
LoRA (Low-Rank Adaptation): Applied to reduce trainable parameters, with configuration:
Rank (r): 4
LoRA Alpha: 8
Dropout: 0.05
Task Type: Causal Language Modeling


Device: Utilized CUDA with automatic device mapping for efficient computation.

Training Setup

Training Arguments:
Output directory: ./results
Per-device train batch size: 2
Gradient accumulation steps: 8
Evaluation strategy: Per epoch
Save strategy: Per epoch
Learning rate: 3e-4
Weight decay: 0.01
FP16: Enabled
Number of epochs: 1
Logging steps: 10


Trainer: Hugging Face Trainer was used, with training completed in under 15 minutes.
Output: The fine-tuned model and tokenizer were saved to ./fine_tuned_phi2 and zipped for easy sharing.

Dependencies

Libraries: transformers, peft, bitsandbytes, accelerate, datasets
Additional tools: sentence_transformers for cosine similarity, matplotlib for visualization
Environment: Google Colab with GPU support

Performance Analysis
The performance of the fine-tuned model was evaluated using perplexity and cosine similarity metrics, comparing it against the base Phi-2 model.
Perplexity
Perplexity measures the model's uncertainty in predicting the next token, with lower values indicating better performance.

Base Model Perplexity: 9.61
Fine-Tuned Model Perplexity: 3.91
Analysis: The significant reduction in perplexity (from 9.61 to 3.91) indicates that the fine-tuned model is more confident and accurate in generating text for the given prompt. This improvement suggests better alignment with the target task (storytelling) due to fine-tuning on the Alpaca dataset.

Cosine Similarity
Cosine similarity was computed between the embeddings of the base and fine-tuned model outputs for the prompt "Tell me a story about a brave warrior," using the all-MiniLM-L6-v2 SentenceTransformer model.

Cosine Similarity Score: 0.79
Analysis: A cosine similarity of 0.79 indicates that the fine-tuned model's output is fairly similar to the base model's output in terms of semantic content, but with notable improvements. The fine-tuned model likely produces more concise, relevant, and engaging narratives, addressing the base model's tendency toward verbosity and irrelevant text.

Visualization
A bar plot was created to visualize the perplexity comparison:
![image](https://github.com/user-attachments/assets/fc66185e-15ea-48b6-bcc1-b50376cdace6)


Base Model: 9.61 (blue bar)
Fine-Tuned Model: 3.91 (orange bar)
The plot clearly shows the fine-tuned model's superior performance, with a lower perplexity score indicating better text generation quality.

Limitations of the Base Model
The base Phi-2 model exhibited the following limitations, which the fine-tuning process aimed to mitigate:

Verbosity: The base model often generated excessive or irrelevant text, resembling textbook-like responses due to its training on primarily textbook data.
Limited Contextual Relevance: Outputs were sometimes less focused or deviated from the prompt's intent, particularly for creative tasks like storytelling.
Lack of Instruction Fine-Tuning: The base model was not fine-tuned for specific tasks, leading to unreliable responses to nuanced or complex instructions.
Inaccurate Code and Facts: The model occasionally produced incorrect code snippets or factual inaccuracies, requiring manual verification.
Language Limitations: Designed for standard English, it struggled with informal language, slang, or other languages.
Potential Biases and Toxicity: Despite safety-focused training data, the model could reflect societal biases or produce harmful content if explicitly prompted.

Improvements After Fine-Tuning
The fine-tuned model addressed several of these limitations:

Improved Coherence and Relevance: Fine-tuning on the Alpaca dataset enhanced the model's ability to generate concise and contextually appropriate responses, particularly for storytelling tasks.
Reduced Perplexity: The drop from 9.61 to 3.91 indicates improved predictive accuracy and confidence in generating relevant text.
Better Handling of Creative Prompts: The fine-tuned model produces more engaging and focused narratives, as evidenced by the cosine similarity score of 0.79, which suggests retained semantic meaning with improved quality.
Efficient Resource Use: The use of LoRA and 4-bit quantization ensured the fine-tuning process was resource-efficient, making it feasible on limited hardware.

Conclusion
The fine-tuning of the Phi-2 model was conducted to enhance its performance for creative text generation, specifically for storytelling tasks. By leveraging the Alpaca dataset and LoRA, the model achieved a significant reduction in perplexity (from 9.61 to 3.91) and maintained a high cosine similarity (0.79) with the base model's output, indicating improved quality while preserving semantic intent. The fine-tuned model is better suited for generating concise, relevant, and engaging responses, overcoming the base model's limitations in verbosity and contextual relevance. However, users should remain cautious of potential inaccuracies, biases, or toxicity, as the model has not been extensively tested for production use.
