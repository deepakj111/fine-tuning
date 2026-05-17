import torch
from unsloth import FastLanguageModel

# 1. Load the fine-tuned model and tokenizer from HuggingFace Hub
model_name = "deepakj111/medical-qwen2.5-0.5B-lora"
max_seq_length = 1024

print(f"Loading model: {model_name}...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name,
    max_seq_length=max_seq_length,
    dtype=None,
    load_in_4bit=True, # Use 4bit quantization to save memory
)

# 2. Enable native 2x faster inference for Unsloth
FastLanguageModel.for_inference(model)

# 3. Define the prompt format
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
    ).to("cuda")
    
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

# 4. Run inference examples
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
