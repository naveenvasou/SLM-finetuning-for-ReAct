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
import argparse

load_dotenv() 
login(token=os.getenv("HF_TOKEN"))

def setup_logging():
    logging.basicConfig(filename='logs/train_lora.log', level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')
    return logging.getLogger(__name__)

####### LOAD CONFIGURATION #######
def load_config(config_path="configs/config.json"):
    with open("configs/config.json", "r") as f:
        config = json.load(f)
    return config

####### LOADING MODEL AND TOKENIZER ##########
def load_model_and_tokenizer(model_name):
    print("####### LOADING MODEL AND TOKENIZER ##########")
    logger.info("Loading base model: " + MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
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
    return model, tokenizer

####### Load the react_samples.json file ##########
def prepare_dataset(data_path, validation_size, tokenizer):
    dataset = load_dataset("json", data_files=data_path, split="train")

    ####### SPLIT DATASET INTO TRAINING AND VALIDATION #######

    split_dataset = dataset.train_test_split(test_size=validation_size, shuffle=True)
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
    return tokenized_train_dataset, tokenized_val_dataset

####### CONFIGURING LORA ##########    
def setup_lora(model, lora_params):
    config = LoraConfig(**lora_params)
    lora_model = get_peft_model(model, config)
    logger.info("lora model created. Trainable Parameters: "+str(lora_model.print_trainable_parameters()) )
    return lora_model

####### TRAINING SETUP ##########
def train( lora_model, tokenizer, train_dataset, val_dataset, 
          training_params, output_dir="./lora-finetuned-model"):
    
    BATCH_SIZE = training_params["batch_size"]
    LR = training_params["learning_rate"]
    NUM_EPOCHS = training_params["num_epochs"]
    SCHEDULER_WARMUP_STEPS = training_params["scheduler_warmup_steps"]
    GRADIENT_CLIP_VAL = training_params["gradient_clip_val"]
    
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=BATCH_SIZE, collate_fn=data_collator)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, collate_fn=data_collator)

    optimizer = AdamW(lora_model.parameters(), lr=LR)

    checkpoint_dir = "./checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    lora_model.train()
    num_training_steps = NUM_EPOCHS * len(train_dataloader)
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=SCHEDULER_WARMUP_STEPS,
        num_training_steps=num_training_steps,
    )

    best_val_loss = float("inf")
    best_model_path = os.path.join("./lora-finetuned-model", "best_model")

    ####### TRAINING LOOP ##########
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
            lr_scheduler.step()
            
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
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            logger.info(f"📈 New best model found! Saving to {best_model_path}")
            lora_model.save_pretrained(best_model_path)
        lora_model.train()
        
    print("\n✅ Training Complete!")
    logger.info("\n✅ Training Complete!")

def main():
    """Parses arguments and orchestrates the training process."""
    parser = argparse.ArgumentParser(
        description="Fine-tune a language model with LoRA."
    )
    # CLI arguments for overriding config values
    parser.add_argument(
        "--model_id",
        type=str,
        default=None,
        help="Hugging Face model ID to use.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Training batch size.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=None,
        help="Learning rate for the optimizer.",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=None,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--validation_split_size",
        type=float,
        default=None,
        help="Fraction of the dataset to use for validation.",
    )
    parser.add_argument(
        "--lora_r",
        type=int,
        default=None,
        help="LoRA `r` parameter.",
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=None,
        help="LoRA `lora_alpha` parameter.",
    )

    args = parser.parse_args()

    # Load and override config
    config = load_config()
    if args.model_id:
        config["model_params"]["model_id"] = args.model_id
    if args.batch_size:
        config["training_params"]["batch_size"] = args.batch_size
    if args.learning_rate:
        config["training_params"]["learning_rate"] = args.learning_rate
    if args.num_epochs:
        config["training_params"]["num_epochs"] = args.num_epochs
    if args.validation_split_size:
        config["data_params"]["validation_split_size"] = args.validation_split_size
    if args.lora_r:
        config["lora_params"]["r"] = args.lora_r
    if args.lora_alpha:
        config["lora_params"]["lora_alpha"] = args.lora_alpha

    # Setup environment
    load_dotenv()
    login(token=os.getenv("HF_TOKEN"))

    global logger
    logger = setup_logging()

    print("####### STARTING TRAINING PROCESS ##########")
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(config["model_params"]["model_id"])

    # Prepare data
    tokenized_train_dataset, tokenized_val_dataset = prepare_dataset(
        data_path="data/react_samples.json",
        validation_size=config["data_params"]["validation_split_size"],
        tokenizer=tokenizer,
    )

    # Set up LoRA
    lora_model = setup_lora(model, config["lora_params"])

    # Start training
    train(
        lora_model,
        tokenizer,
        tokenized_train_dataset,
        tokenized_val_dataset,
        config["training_params"],
    )

if __name__ == "__main__":
    main()