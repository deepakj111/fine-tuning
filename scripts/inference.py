import logging

import torch
from unsloth import FastLanguageModel

# Configure structured logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# 1. Load the fine-tuned model and tokenizer from HuggingFace Hub
model_name = "deepakj111/medical-qwen2.5-0.5B-lora"
max_seq_length = 1024

logger.info(f"Loading model: {model_name}...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name,
    max_seq_length=max_seq_length,
    dtype=None,
    load_in_4bit=True,  # Use 4bit quantization to save memory
)

# 2. Enable native 2x faster inference for Unsloth
FastLanguageModel.for_inference(model)


# 3. Define the prompt format
def generate_response(question: str) -> str:
    # Use tokenizer.apply_chat_template for standard prompt formatting
    messages = [
        {"role": "system", "content": "You are a helpful medical assistant."},
        {"role": "user", "content": question},
    ]

    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to("cuda")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            do_sample=True,  # Set to True for more creative/diverse answers, False for greedy
            temperature=0.3,  # Low temperature for more factual responses
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
        "What is the difference between ibuprofen and acetaminophen?",
    ]

    logger.info("\n" + "=" * 60)
    logger.info("Starting Medical Inference Examples")
    logger.info("=" * 60)

    for i, question in enumerate(example_questions, 1):
        logger.info(f"\n[Example {i}]")
        logger.info(f"Question: {question}")
        logger.info("-" * 60)

        answer = generate_response(question)

        logger.info(f"Answer:   {answer}")
        logger.info("=" * 60)
