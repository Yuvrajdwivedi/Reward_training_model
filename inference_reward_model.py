import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

# 1. Paths
base_model_path = r"C:\QwenModel"
adapter_path = "./reward_model_output/checkpoint-1215" # Pointing directly to your final trained adapter!

# 2. Recreate the 4-bit config so it fits on your RTX 4050 safely
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16
)

print("Loading base model in 4-bit...")
base_model = AutoModelForSequenceClassification.from_pretrained(
    base_model_path,
    num_labels=1,
    device_map="auto",
    local_files_only=True,
    quantization_config=bnb_config
)
tokenizer = AutoTokenizer.from_pretrained(base_model_path, local_files_only=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 3. Snap your trained LoRA adapter onto the base model
print("Attaching your trained reward adapter (Checkpoint 1215)...")
model = PeftModel.from_pretrained(base_model, adapter_path)
model.eval() # Set the model to evaluation (testing) mode

# 4. Create a function to score text
def get_score(user_prompt, AI_response):
    # Combine the prompt and response exactly how the model expects it
    text = f"User: {user_prompt}\nAssistant: {AI_response}"
    
    # Convert text to numbers and send to GPU
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to("cuda")
    
    # Run the model without calculating gradients (saves memory)
    with torch.no_grad():
        outputs = model(**inputs)
        
    # Extract the single scalar score
    score = outputs.logits[0][0].item()
    return score

#===============prompt and responses for testing===================
prompt = "Write a haiku about the ocean."

good_response = "Blue waves crash the shore,\nSalty breeze blows through the air,\nOcean deep and wide."

bad_response = "The ocean is a massive body of saltwater that covers most of the Earth. It is very deep and contains lots of fish."

print("\nEvaluating responses...")
score1 = get_score(prompt, good_response)
score2 = get_score(prompt, bad_response)

print(f"\nScore for GOOD response: {score1:.4f}")
print(f"Score for BAD response:  {score2:.4f}")

if score1 > score2:
    print("\n✅ Success! Your model correctly prefers the helpful answer.")
else:
    print("\n❌ The model got confused on this specific prompt.")