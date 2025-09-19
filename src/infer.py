import json
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import logging
import os
from glob import glob
import re
import argparse
import torch

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)

####### LOAD CONFIGURATION #######

with open("configs/config.json", "r") as f:
    config = json.load(f)
MODEL_NAME = config["model_params"]["model_id"]   


def load_model_and_tokenizer(lora_checkpoint_path):

    model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            trust_remote_code=True,
            attn_implementation='eager'
    )
    model.config.use_cache = False
    LoRA_adapter_checkpoint = lora_checkpoint_path
    model = PeftModel.from_pretrained(model, LoRA_adapter_checkpoint)
    model = model.merge_and_unload()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    logger.info("Instantiated the model and tokenizer")
    return model, tokenizer

def generate_response(prompt: str, max_new_tokens: str = 128) -> str:
    if os.path.isdir("./lora-finetuned-model/best_model"):
        lora_checkpoint_path = "./lora-finetuned-model/best_model/"
    else:
        lora_checkpoint_path = sorted(glob("./checkpoints/epoch_*"))[-1]
    logger.info(f"Loading model and tokenizer from {lora_checkpoint_path}")
    model, tokenizer = load_model_and_tokenizer(lora_checkpoint_path)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    logger.info("Generating text...")
    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens,
    eos_token_id=tokenizer.eos_token_id,
    do_sample=False)
    logger.info("text generation done...")
    generated_text = tokenizer.decode(outputs[0][len(inputs["input_ids"][0]):], skip_special_tokens=True)
    return generated_text.strip()

def parse_generated_text(text: str):

    thought_match = re.search(r"Thought:\s*(.*)", text)
    action_match = re.search(r"Action:\s*(.*)", text)
    
    thought = thought_match.group(1).strip() if thought_match else None
    action = action_match.group(1).strip() if action_match else None
    
    return thought, action

def simulate_action_execution(action: str) -> str:

    print(f"Executing: {action}")
    
    # Simple simulation for a search action
    if action.lower().startswith("search['president of france']"):
        return "Emmanuel Macron"
    # Simulation for your example: "president of France squared"
    elif action.lower() == "search['current president of france']":
         return "Emmanuel Macron"
    # Add more simulation logic here for other actions (e.g., Calculator)
    else:
        return "Could not find relevant information."

def main(question: str):

    prompt1 = f"Question: {question}\nThought:"
    logger.info("Generating text....")
    generated_text1 = generate_response(prompt1)
    
    thought, action = parse_generated_text(f"Thought: {generated_text1}")

    if not thought or not action:
        print("Error: Could not parse Thought or Action from model output.")
        print(f"Raw output: {generated_text1}")
        return

    observation = simulate_action_execution(action)

    prompt2 = (
        f"Question: {question}\n"
        f"Thought: {thought}\n"
        f"Action: {action}\n"
        f"Observation: {observation}\n"
        f"Final Answer:"
    )
    final_answer = generate_response(prompt2)

    
    print(f"Thought: {thought}")
    print(f"Action: {action}")
    print(f"Observation: {observation}")
    print(f"Final Answer: {final_answer}")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a one-cycle ReAct loop with a fine-tuned model.")
    parser.add_argument("--question", type=str, required=True, help="The question to ask the model.")
    args = parser.parse_args()
    
    main(args.question)