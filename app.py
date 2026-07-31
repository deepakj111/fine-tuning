import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Configure Streamlit page
st.set_page_config(page_title="Medical Q&A Assistant", page_icon="🏥", layout="wide")

st.title("🏥 Medical Q&A Assistant")
st.markdown("*Powered by Fine-tuned Qwen2.5-0.5B (Running natively on CPU)*")

# Sidebar for controls
with st.sidebar:
    st.header("⚙️ Model Settings")
    model_choice = st.radio(
        "Choose Model Version:",
        ("Fine-Tuned (Medical)", "Base Model (General)"),
        help="Compare how the base model behaves versus the medical fine-tuned version.",
    )

    st.header("💡 Example Prompts")
    examples = [
        "What are the classic symptoms of Type 2 Diabetes?",
        "How do you treat a mild fever in adults?",
        "Explain the difference between a virus and a bacteria.",
        "What is hypertension and how is it managed?",
    ]

    # We use session state to populate chat input from buttons
    for ex in examples:
        if st.button(ex, use_container_width=True):
            st.session_state.prompt_text = ex
            st.rerun()


# Load model using Streamlit cache so it only loads once
@st.cache_resource(show_spinner=False)
def load_models():
    base_model_id = "Qwen/Qwen2.5-0.5B-Instruct"
    lora_id = "deepakj111/medical-qwen2.5-0.5B-lora"

    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    # CPU-friendly loading in bfloat16
    base_model = AutoModelForCausalLM.from_pretrained(base_model_id, device_map="cpu", torch_dtype=torch.bfloat16)

    # Wrap with PEFT adapter (fine-tuned weights)
    model = PeftModel.from_pretrained(base_model, lora_id)
    return model, tokenizer


try:
    with st.spinner("Loading model weights (approx 1GB)... This may take 30-60 seconds on first load."):
        model, tokenizer = load_models()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()


# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Determine the prompt
user_prompt = st.chat_input("Ask a medical question...")
if "prompt_text" in st.session_state and st.session_state.prompt_text:
    user_prompt = st.session_state.prompt_text
    st.session_state.prompt_text = None  # Clear after use

# Accept user input
if user_prompt:
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": user_prompt})

    with st.chat_message("user"):
        st.markdown(user_prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        message_placeholder = st.empty()

        # Prepare ChatML template
        chat_messages = [{"role": "system", "content": "You are a helpful medical assistant."}]
        for msg in st.session_state.messages:
            chat_messages.append({"role": msg["role"], "content": msg["content"]})

        formatted_prompt = tokenizer.apply_chat_template(chat_messages, tokenize=False, add_generation_prompt=True)

        try:
            tokenizer.truncation_side = "left"
            inputs = tokenizer(formatted_prompt, return_tensors="pt", truncation=True, max_length=768)
            # Ensure tensors are on CPU
            inputs = {k: v.to("cpu") for k, v in inputs.items()}

            with st.spinner(f"Generating answer using {model_choice}... (CPU inference is slower)"):
                from contextlib import nullcontext

                ctx = model.disable_adapter() if model_choice == "Base Model (General)" else nullcontext()

                with torch.no_grad(), ctx:
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=256,
                        do_sample=True,
                        temperature=0.3,
                        repetition_penalty=1.15,
                        pad_token_id=tokenizer.eos_token_id,
                    )

                input_len = inputs["input_ids"].shape[1]
                response = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True).strip()
                message_placeholder.markdown(response)

            st.session_state.messages.append({"role": "assistant", "content": response})
        except Exception as e:
            st.error(f"Generation failed: {e}")
