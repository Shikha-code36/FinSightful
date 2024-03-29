import os
import streamlit as st
import time
import langchain
from langchain.llms import OpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chains.qa_with_sources.loading import load_qa_with_sources_chain
from langchain_community.vectorstores.faiss import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import SeleniumURLLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from dotenv import load_dotenv

load_dotenv()  # take environment variables from .env (especially openai api key)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

st.title("FinSightFul: News Research Tool 📈")
st.sidebar.title("News Article URLs")

urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)

process_url_clicked = st.sidebar.button("Process URLs")

main_placeholder = st.empty()
llm = OpenAI(temperature=0.9, max_tokens=500) 

if 'query' not in st.session_state:
    st.session_state.query = ''

if 'result' not in st.session_state:
    st.session_state.result = None

if process_url_clicked:
    #LOAD DATA
    loader = SeleniumURLLoader(urls=urls)
    main_placeholder.text("Data Loading...Started...✅✅✅")
    data = loader.load()
    # split data
    text_splitter = RecursiveCharacterTextSplitter(
        separators=['\n\n', '\n', '.', ','],
        chunk_size=1000
    )
    main_placeholder.text("Text Splitter...Started...✅✅✅")
    docs = text_splitter.split_documents(data)

    # create embeddings and save it to FAISS index
    embeddings = OpenAIEmbeddings()
    vectorstore_openai = FAISS.from_documents(docs, embeddings)
    main_placeholder.text("Embedding Vector Started Building...✅✅✅")
    time.sleep(2)

    # Save the FAISS index to a pickle file
    vectorstore_openai.save_local("faiss_store")

st.session_state.query = main_placeholder.text_input("Question: ", value=st.session_state.query)
submit_button = st.button('Submit')

if submit_button:
    if st.session_state.query:
        #st.write("new query:", st.session_state.query)
        embeddings = OpenAIEmbeddings()
        llm = OpenAI(temperature=0.9, max_tokens=500)
        vectorstore = FAISS.load_local("faiss_store", embeddings, allow_dangerous_deserialization=True)
        chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorstore.as_retriever())
        st.session_state.result = chain({"question": st.session_state.query}, return_only_outputs=True)
        #st.write("res", st.session_state.result)

        if st.session_state.result and st.session_state.result["answer"]:
            st.header("Answer")
            st.write(st.session_state.result["answer"])

            # Display sources, if available
            sources = st.session_state.result.get("sources", "")
            if sources:
                st.subheader("Sources:")
                sources_list = sources.split("\n")  # Split the sources by newline
                for source in sources_list:
                    st.write(source)
        else:
            st.write("No answer found.")
