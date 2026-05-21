from utils.data_utils import format_medical_sample, format_ultrachat


def test_format_medical_sample():
    row = {"question": "What is the common cold?", "answer": "A viral infection of your nose and throat."}

    formatted = format_medical_sample(row)

    assert "text" in formatted
    text = formatted["text"]

    # Check if correct tokens are present
    assert "<|im_start|>system" in text
    assert "<|im_start|>user" in text
    assert "<|im_start|>assistant" in text

    # Check if content is present
    assert "What is the common cold?" in text
    assert "A viral infection of your nose and throat." in text


def test_format_ultrachat():
    example = {"messages": [{"role": "user", "content": "Hello!"}, {"role": "assistant", "content": "Hi there!"}]}

    formatted = format_ultrachat(example)

    assert "text" in formatted
    text = formatted["text"]

    assert "<|im_start|>user\nHello!\n<|im_end|>" in text
    assert "<|im_start|>assistant\nHi there!\n<|im_end|>" in text


def test_format_ultrachat_empty_messages():
    example = {"messages": []}
    formatted = format_ultrachat(example)
    assert formatted["text"] == ""
