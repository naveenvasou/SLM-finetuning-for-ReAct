from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from transformers import TrainingArguments, Trainer, BitsAndBytesConfig, DataCollatorForLanguageModeling, get_linear_schedule_with_warmup
import logging
import torch
from datasets import Dataset, load_dataset
from torch.utils.data import DataLoader
from dotenv import load_dotenv
import os
from tqdm import tqdm
import json
from torch.optim import AdamW
from huggingface_hub import login

load_dotenv() 
login(token=os.getenv("HF_TOKEN"))

logging.basicConfig(filename='logs/train_lora.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)

####### LOAD CONFIGURATION #######

with open("configs/config.json", "r") as f:
    config = json.load(f)
    
MODEL_NAME = config["model_params"]["model_id"]
BATCH_SIZE = config["training_params"]["batch_size"]
LR = config["training_params"]["learning_rate"]
NUM_EPOCHS = config["training_params"]["num_epochs"]
SCHEDULER_WARMUP_STEPS = config["training_params"]["scheduler_warmup_steps"]
GRADIENT_CLIP_VAL = config["training_params"]["gradient_clip_val"]
VALIDATION_SIZE = config["data_params"]["validation_split_size"]

####### LOADING MODEL AND TOKENIZER ##########
print("####### LOADING MODEL AND TOKENIZER ##########")
logger.info("Loading base model: " + MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
)
model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation='eager'
)
model.config.use_cache = False
logger.info("Instantiated the model")
logger.info("Loading tokenizer for base model")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"
logger.info("Instantiated the tokenizer")
print("####### INSTANTIATED MODEL AND TOKENIZER ##########")

####### Load the react_samples.json file ##########

dataset = load_dataset("json", data_files="data/react_samples.json", split="train")

####### SPLIT DATASET INTO TRAINING AND VALIDATION #######

split_dataset = dataset.train_test_split(test_size=VALIDATION_SIZE, shuffle=True)
train_dataset = split_dataset["train"]
val_dataset = split_dataset["test"]

####### Prepare data for causal language modeling #######
print("####### Prepare data for causal language modeling ##########")
def format_and_tokenize(example):
    formatted_prompt = (
        f"Question: {example['question']}\nThought: {example['thought']}\n"
        f"Action: {example['action']}\nObservation: {example['observation']}\n"
        f"Final Answer: {example['final_answer']}"
    )
    return tokenizer(
        formatted_prompt + tokenizer.eos_token,
        truncation=True,
        max_length=512,
    )
    
tokenized_train_dataset = train_dataset.map(format_and_tokenize, remove_columns=train_dataset.column_names)
tokenized_val_dataset = val_dataset.map(format_and_tokenize, remove_columns=val_dataset.column_names)

logger.info("Data Loaded and Tokenized")
print("####### Data Loaded and Tokenized ##########")


####### CONFIGURING LORA ##########    

config = LoraConfig(**config["lora_params"])

lora_model = get_peft_model(model, config)
logger.info("lora model created. Trainable Parameters: "+str(lora_model.print_trainable_parameters()) )

####### DATALOADER SETUP ##########

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
train_dataloader = DataLoader(
    tokenized_train_dataset, shuffle=True, batch_size=BATCH_SIZE, collate_fn=data_collator
)
val_dataloader = DataLoader(
    tokenized_val_dataset, batch_size=BATCH_SIZE, collate_fn=data_collator
)


####### TRAINING LOOP ##########

optimizer = AdamW(lora_model.parameters(), lr=LR)

checkpoint_dir = "./checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)
lora_model.train()
num_training_steps = NUM_EPOCHS * len(train_dataloader)
#lr_scheduler = get_linear_schedule_with_warmup(
#    optimizer=optimizer,
#    num_warmup_steps=SCHEDULER_WARMUP_STEPS,
#    num_training_steps=num_training_steps,
#)

best_val_loss = float("inf")
best_model_path = os.path.join("./lora-finetuned-model", "best_model")

for epoch in range(NUM_EPOCHS):
    logger.info(f"\n--- Epoch {epoch + 1}/{NUM_EPOCHS} ---")
    print(f"\n--- Epoch {epoch + 1}/{NUM_EPOCHS} ---\n")
    total_train_loss = 0
    train_progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1} Training")
    
    for batch in train_progress_bar:
        optimizer.zero_grad()
        
        batch = {k: v.to(lora_model.device) for k, v in batch.items()}
        ### FORWARD PASS ###
        outputs = lora_model(**batch)
        ### LOSS COMPUTATION ###
        loss = outputs.loss
        ### BACKPROPAGATION ###
        loss.backward()
        ### GRADIENT CLIPPING ###
        torch.nn.utils.clip_grad_norm_(lora_model.parameters(), GRADIENT_CLIP_VAL)
        ### OPTIMIZER ###
        optimizer.step()
        ### SCHEDULER ###
        #lr_scheduler.step()
        
        train_progress_bar.set_postfix({"loss": loss.item()})
        total_train_loss += loss.item()
    
    avg_loss = total_train_loss / len(train_dataloader)
    print(f"\n  Average Loss for Epoch {epoch + 1}: {avg_loss:.4f}")
    logger.info(f"\n  Average Loss for Epoch {epoch + 1}: {avg_loss:.4f}")
    
    checkpoint_path = os.path.join(checkpoint_dir, f"epoch_{epoch+1}")
    lora_model.save_pretrained(checkpoint_path)
    logger.info(f"\n LoRA adapter checkpoint saved to {checkpoint_path}")
    
    ### EVALUATION  LOOP ###
    lora_model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for batch in tqdm(val_dataloader, desc=f"Epoch {epoch+1} Validation"):
            batch = {k:v.to(lora_model.device) for k, v in batch.items()}
            outputs = lora_model(**batch)
            total_val_loss += outputs.loss.item()
    avg_val_loss = total_val_loss / len(val_dataloader)
    print(f"\n  Average Validation Loss: {avg_val_loss:.4f}")
    logger.info(f"\n  Average Validation Loss: {avg_val_loss:.4f}")
    
    ####### TRAINING LOOP ##########
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        logger.info(f"📈 New best model found! Saving to {best_model_path}")
        lora_model.save_pretrained(best_model_path)
    
    
print("\n✅ Training Complete!")
logger.info("\n✅ Training Complete!")

