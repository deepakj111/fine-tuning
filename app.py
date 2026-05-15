import streamlit as st
import torch
from unsloth import FastLanguageModel

# Configure Streamlit page
st.set_page_config(
    page_title="Medical Q&A Assistant",
    page_icon="🏥",
    layout="centered"
)

st.title("🏥 Medical Q&A Assistant")
st.markdown("*Powered by Fine-tuned Qwen2.5-0.5B via Unsloth & QLoRA*")

# Load model using Streamlit cache so it only loads once
@st.cache_resource
def load_model():
    model_name = "deepakj111/medical-qwen2.5-0.5B-lora"
    max_seq_length = 1024

    # Using st.spinner inside cache requires the user to see the first load
    # but subsequent reloads will skip this.
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)
    return model, tokenizer

try:
    with st.spinner("Loading model weights from Hugging Face... This may take a minute."):
        model, tokenizer = load_model()
except Exception as e:
    st.error(f"Error loading model. Please ensure you have a GPU with sufficient memory. Details: {e}")
    st.stop()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("Ask a medical question... (e.g., What are the symptoms of Type 2 Diabetes?)"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        
        # Build prompt using ChatML format (matching inference.py)
        formatted_prompt = (
            f"<|im_start|>system\nYou are a helpful medical assistant.\n<|im_end|>\n"
            f"<|im_start|>user\n{prompt}\n<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        
        try:
            inputs = tokenizer(
                formatted_prompt, 
                return_tensors="pt", 
                truncation=True, 
                max_length=512
            ).to("cuda")
            
            with st.spinner("Generating answer..."):
                with torch.no_grad():
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
                
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": response})
        except Exception as e:
            st.error(f"Generation failed: {e}")
