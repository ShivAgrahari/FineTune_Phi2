# Fine-Tuning Phi-2 for Enhanced Storytelling

This project provides a tutorial on **parameter-efficient fine-tuning (PEFT)** and **quantization** of the Phi-2 model to improve its storytelling capabilities.

We use **LoRA (Low-Rank Adaptation)** for PEFT and **4-bit quantization** to compress the model, fine-tuning it on the [`tatsu-lab/alpaca`](https://huggingface.co/datasets/tatsu-lab/alpaca) dataset to enhance its ability to generate coherent and engaging narratives — such as responses to prompts like:

> _"Tell me a story about a brave warrior."_

Refer to the [Phi-2 model page](https://huggingface.co/microsoft/phi-2) and dataset page for additional details.

---

## Usage

Start by cloning the repository, setting up a conda environment, and installing the dependencies.

>  Tested with Python 3.9 and CUDA 11.7

\`\`\`bash
git clone https://github.com/ShivAgrahari/FineTune_Phi2.git
cd FineTune_Phi2

conda create -n llm python=3.9
conda activate llm

pip install -r requirements.txt
\`\`\`

---

## Requirements

The following dependencies are listed in \`requirements.txt\`:

\`\`\`txt
bitsandbytes==0.40.2
datasets==3.3.2
huggingface-hub==0.29.1
matplotlib==3.8.2
peft==0.4.0
torch==2.6.0
transformers==4.31.0
\`\`\`

---

##  Fine-Tuning the Model

To fine-tune the Phi-2 model on the \`tatsu-lab/alpaca\` dataset (or any dataset with similar instruction-output structure), run:

\`\`\`bash
accelerate config default

python Fine_Tuning_Phi2.py \\
  --dataset="tatsu-lab/alpaca" \\
  --base_model="microsoft/phi-2" \\
  --model_name="fine_tuned_phi2" \\
  --auth_token=<HF_AUTH_TOKEN> \\
  --push_to_hub
\`\`\`

> ⚠️ Make sure you have a Hugging Face Hub token if you're using a private dataset.

> ⏱ Fine-tuning takes approximately **15 minutes** on a **single A100 GPU** with default settings.

---

###  Notes

- The script uses **LoRA** for efficient fine-tuning and **4-bit quantization** to reduce memory usage.
- Only the **LoRA parameters** are saved after training. These are loaded onto the base Phi-2 model for inference.
- The fine-tuned model is saved to \`./fine_tuned_phi2\` and can optionally be pushed to Hugging Face Hub using the \`--push_to_hub\` flag.
- Ensure the dataset contains **instruction** and **output** fields, as required by the tokenization logic.

---

##  Performance Evaluation

The fine-tuning script includes evaluation using:

- **Perplexity** (lower is better)
- **Cosine Similarity** (higher is better)

### Results:
![image](https://github.com/user-attachments/assets/6e73bc32-88e1-4c98-8d62-ca07f8a5199b)


| Metric              | Base Model | Fine-Tuned |
|---------------------|------------|-------------|
| Perplexity          | 9.61       | 3.91        |
| Cosine Similarity   | —          | 0.79        |

This indicates improved output quality while retaining semantic intent.


---

##  Final Notes

To run the evaluation and inference:

- Make sure the fine-tuned model is available **locally** or via the **Hugging Face Hub**.
- Modify prompt inputs and generation parameters as needed to test storytelling capabilities.
