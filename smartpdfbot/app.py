from rag_component import rag_function
import streamlit as st
from ingest_gemini import extract_text_from_pdf,create_chunks, store_into_vectordb,delete

st.title("SMART PDF BOT")

with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                extracted_text = extract_text_from_pdf(pdf_docs)
                created_chunks = create_chunks(extracted_text)
                store_into_vectordb(created_chunks)
                st.success("Done")

# set initial message
if "messages" not in st.session_state.keys():
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello there, how can I help you?"}
    ]

# display messages
if "messages" in st.session_state.keys():
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

# get user input
user_prompt = st.chat_input()

if user_prompt is not None:
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.write(user_prompt)

if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Loading..."):
            ai_response = rag_function(user_prompt)
            st.write(ai_response)
            
    new_ai_message = {"role": "user", "content": ai_response}
    st.session_state.messages.append(new_ai_message)


