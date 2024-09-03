__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
from embedding_generator import EmbeddingGenerator
from chatbot_qa import ChatbotQA
from langchain_community.callbacks.streamlit import (
    StreamlitCallbackHandler,
)
import streamlit as st


def main():
    # Step 2: Generate Embeddings
    generator = EmbeddingGenerator()
    full_text = generator.load_papers()
    paper_chunks = generator.split_text(full_text)
    retriever = generator.generate_embeddings(paper_chunks)
    print("Embeddings generated successfully.")
    # Step 3: Initialize and Run Chatbot
    chatbot = ChatbotQA(retriever)
    chatbot.setup_chain()
    st_callback = StreamlitCallbackHandler(st.container())
   #st.title ("Car parts user recomendation system (Powered by OpenAI )")
    
    st.markdown(
        """
        <h1 style='font-size:25px;'>
        Car Parts User Recommendation System (Powered by OpenAI)
        </h1>
        """,
        unsafe_allow_html=True
    )
    if prompt := st.chat_input():
        st.chat_message("user").write(prompt)
        with st.chat_message("assistant"):
            st_callback = StreamlitCallbackHandler(st.container())
            response = chatbot.answer_question(prompt, st_callback)
            st.write(response)
    # while True:
    #     user_input = input("Human: ")
    #     if user_input.lower() == 'exit':
    #         print("Exiting chatbot. Goodbye!")
    #         break
    #     else:
    #         response = chatbot.answer_question(user_input)
    #         print("AI Assistant:", response)
if __name__ == "__main__":
    main()
