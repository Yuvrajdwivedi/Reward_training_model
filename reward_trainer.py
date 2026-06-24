import json 
import torch
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, BitsAndBytesConfig
from trl import RewardTrainer, RewardConfig
from huggingface_hub import snapshot_download
from peft import LoraConfig, TaskType

# 1. Point this directly to your local directory path

model_path = r"C:\QwenModel"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",           # NormalFloat4 (best for QLoRA)
    bnb_4bit_use_double_quant=True,      # Saves even more memory
    bnb_4bit_compute_dtype=torch.bfloat16
)
# 2. Load your dataset locally 
# (If you have a local CSV/JSON instead of Hugging Face Hub)
# raw_dataset = load_dataset("json", data_files="local_data.json", split="train[:10%]")
raw_dataset = load_dataset("parquet", data_files="train-00000-of-00001.parquet", split="train[:10%]")
print("YOUR DATA COLUMNS ARE:", raw_dataset.column_names)

# 3. Clean the columns (RewardTrainer directly uses 'chosen' and 'rejected')
# We do NOT use make_conversation here because your dataset already has these columns.
processed_dataset = raw_dataset.select_columns(["chosen", "rejected"])

# Split dataset into train (90%) and evaluation (10%)
dataset_split = processed_dataset.train_test_split(test_size=0.1)
train_dataset = dataset_split["train"]
eval_dataset = dataset_split["test"]

# 4. Load from Local Storage
# Because 'model_path' points to a local directory, no APIs or remotes are touched.
# Fallback logic handles downloading directly through Python on the first run if directory is empty.
try:
    print("Attempting to load model locally...")
    model = AutoModelForSequenceClassification.from_pretrained( #this is used to load the model from the local path
        model_path,
        num_labels=1, # We want a single scalar reward output
        torch_dtype="auto", # Let Hugging Face decide the best dtype (e.g., bf16 if supported)
        device_map="auto", # Automatically place model on GPU if available
        local_files_only=True, # Forces HF to crash if it tries to touch the internet
        quantization_config=bnb_config
    )
except (OSError, ValueError):
    print("Model not found locally. Downloading from Hugging Face directly to C:\\QwenModel...")

    # Safely downloads the files to your hard drive WITHOUT loading them into GPU memory first
    snapshot_download(repo_id="Qwen/Qwen2.5-1.5B", local_dir=model_path, token=False)

    model = AutoModelForSequenceClassification.from_pretrained(
        "Qwen/Qwen2.5-1.5B",
        num_labels=1,
        torch_dtype="auto", 
        device_map="auto",
        token=False,
        quantization_config=bnb_config
    )
    model.save_pretrained(model_path)

try:
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
except (OSError, ValueError):
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B")
    tokenizer.save_pretrained(model_path)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token # Reward models often use the same token for padding and end-of-sequence to avoid confusion during training.

peft_config = LoraConfig(
    task_type=TaskType.SEQ_CLS, 
    r=16,                       
    lora_alpha=32,              
    lora_dropout=0.05,
    bias="none",
    target_modules=["q_proj", "v_proj"] 
)

# 5. Config & Trainer
training_args = RewardConfig(
    output_dir="./reward_model_output", 
    logging_steps=10, # Log every 10 steps to see training progress
    num_train_epochs=3, # Train for 3 epochs (adjust as needed)
    per_device_train_batch_size=1,# Keep this small if your GPU memory is limited, especially with large models
    gradient_checkpointing=True,       
    gradient_accumulation_steps=8, # Accumulate gradients over 4 steps to effectively have a batch size of 8 without OOM errors
    max_length=512, # Truncate conversations to 1024 tokens (adjust based on your model's max input size)
    bf16=True, # Use bf16 if your GPU supports it for faster training and reduced memory usage
    eval_strategy="steps", # Evaluate every 100 steps to monitor performance on the validation set
    eval_steps=100, # Evaluate every 100 steps to monitor performance on the validation set
    save_strategy="steps", # Save model checkpoints every 100 steps to avoid losing progress
    save_steps=100, # Save model checkpoints every 100 steps to avoid losing progress
    logging_strategy="steps",# Log every 10 steps to see training progress
)

trainer = RewardTrainer(
    model=model,   # Your loaded local model              
    processing_class=tokenizer,     # Your loaded local tokenizer    
    args=training_args, # Training arguments defined above
    train_dataset=train_dataset, # Training dataset (90% of your data)
    eval_dataset=eval_dataset,   # Evaluation dataset (10% of your data)
    peft_config=peft_config,
)

trainer.train()