import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# 1. Load the fine-tuned model and tokenizer from HuggingFace Hub
base_model_name = "Qwen/Qwen2.5-0.5B-Instruct"
adapter_name = "deepakj111/medical-qwen2.5-0.5B-lora"

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Loading model on {device.upper()}...")

# Load base model
model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    device_map=device,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
)

# Apply LoRA adapters
model = PeftModel.from_pretrained(model, adapter_name)
tokenizer = AutoTokenizer.from_pretrained(base_model_name)

# 2. Define the prompt format
def generate_response(question: str) -> str:
    # Using ChatML format which Qwen2.5-Instruct uses
    prompt = (
        f"<|im_start|>system\nYou are a helpful medical assistant.\n<|im_end|>\n"
        f"<|im_start|>user\n{question}\n<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    
    inputs = tokenizer(
        prompt, 
        return_tensors="pt", 
        truncation=True, 
        max_length=512
    ).to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=200,
            do_sample=True, # Set to True for more creative/diverse answers, False for greedy
            temperature=0.3, # Low temperature for more factual responses
            repetition_penalty=1.15,
            pad_token_id=tokenizer.eos_token_id,
        )
        
    # Extract only the newly generated tokens
    input_len = inputs["input_ids"].shape[1]
    response = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True).strip()
    return response

# 3. Run inference examples
if __name__ == "__main__":
    example_questions = [
        "What are the common symptoms of Type 2 Diabetes?",
        "How should one treat a mild burn at home?",
        "What is the difference between ibuprofen and acetaminophen?"
    ]
    
    print("\n" + "="*60)
    print("Starting Medical Inference Examples")
    print("="*60)
    
    for i, question in enumerate(example_questions, 1):
        print(f"\n[Example {i}]")
        print(f"Question: {question}")
        print("-" * 60)
        
        answer = generate_response(question)
        
        print(f"Answer:   {answer}")
        print("="*60)
