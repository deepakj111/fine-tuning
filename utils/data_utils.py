def format_medical_sample(row: dict) -> dict:
    return {
        "text": (
            f"<|im_start|>system\n"
            f"You are a medical education assistant. "
            f"Answer the following medical question clearly and accurately.\n"
            f"<|im_end|>\n"
            f"<|im_start|>user\n{row['question']}\n<|im_end|>\n"
            f"<|im_start|>assistant\n{row['answer']}<|im_end|>"
        )
    }


def format_ultrachat(example: dict) -> dict:
    messages = example.get("messages", [])
    text = ""
    for msg in messages[:4]:
        role = msg.get("role", "user")
        content = msg.get("content", "")[:300]
        text += f"<|im_start|>{role}\n{content}\n<|im_end|>\n"
    return {"text": text.strip()}
