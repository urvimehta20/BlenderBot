import streamlit as st
from streamlit_chat import message as st_message
from transformers import BlenderbotTokenizer
from transformers import BlenderbotForConditionalGeneration


@st.cache_resource
def get_models():
    # it may be necessary for other frameworks to cache the model
    # seems pytorch keeps an internal state of the conversation
    model_name = "facebook/blenderbot-400M-distill"
    tokenizer = BlenderbotTokenizer.from_pretrained(model_name)
    model = BlenderbotForConditionalGeneration.from_pretrained(model_name)
    return tokenizer, model


if "history" not in st.session_state:
    st.session_state.history = []

st.title("GPT")


def generate_answer():
    tokenizer, model = get_models()
    user_message = st.session_state.input_text
    
    knowledge_query = "What is the capital of France?"
    input_text = f"{user_message}\n{knowledge_query}"
    
    inputs = tokenizer(st.session_state.input_text, return_tensors="pt")
    result = model.generate(**inputs)
    message_bot = tokenizer.decode(
        result[0], skip_special_tokens=True
    )  # .replace("<s>", "").replace("</s>", "")

    st.session_state.history.append({"message": user_message, "is_user": True})
    st.session_state.history.append({"message": message_bot, "is_user": False})


st.text_input("Enter a prompt..", key="input_text", on_change=generate_answer)

for i, chat in enumerate(st.session_state.history):
    st_message(**chat, key=str(i)) #unpacking
