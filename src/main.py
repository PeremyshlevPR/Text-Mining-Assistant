import streamlit as st
from typing import List
import logging

from chromadb import Embeddings
from langchain_community.vectorstores import Chroma
from embeddings import TogetherEmbeddings
from translators import YandexTranslator
from llms import YandexGPT
from conf.settings import settings

from logger import init_logs
init_logs()
logger = logging.getLogger(__name__)


_translator = YandexTranslator(settings.YANDEX_FOLDER_ID, settings.YANDEX_OAUTH_TOKEN)


def init_css():
    with open('static/css/styles.css') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    
    
def get_vectorstore(persist_directory: str, embeddings: Embeddings):
    vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    n_docs = len(vectorstore.get(include=["metadatas"])["ids"])
    logger.info(f'Vectorstore loaded. Got {n_docs} documents.')
    return vectorstore


def get_source_names(vectorstore) -> List[str]:
    result = vectorstore.get(include=["metadatas"])
    sources = sorted(list(set([doc['docname'] for doc in result['metadatas']])))
    logger.info(f'Vectorstore has documents from {len(sources)} sources.')
    return sources

def get_relevant_docs(vectorstore, query, sources, k=3):
    filter_ = {
        'docname': {
            '$in': sources
        }
    }
    response = vectorstore.similarity_search_with_score(query, filter=filter_, k=k)
    return [doc for doc, score in response]


def handle_user_input(question, llm, vectorstore, sources, n_docs=3):
    if sources:
        logger.info(f'Soures passed: {", ".join(sources)}')
        en_query = _translator.translate(question, target='en')[0]
        docs = get_relevant_docs(vectorstore, query=en_query, sources=sources, k=n_docs)
        for i, doc in enumerate(docs):
            docs[i].page_content = _translator.translate(doc.page_content, target='ru')[0]
        context = '\n\n'.join([doc.page_content for doc in docs])

        logger.info(f'Documents succesfully retrieved from sources: {", ".join([doc.metadata["docname"] for doc in docs])}')
    else:
        logger.info('')
        docs = []
        context = None

    response = llm(prompt=question, context=context)
    return {
        'llm_output': response,
        'docs': docs
    }
 
    
def main():
    st.set_page_config(page_title="Text Mining",
                       page_icon=":books:")
    init_css()
    
    if 'embeddings' not in st.session_state:
        logger.info('Initializing embeddings...')
        st.session_state.embeddings = TogetherEmbeddings(model=settings.EMBEDDING_MODEL, token=settings.TOGETHER_TOKEN)

    if 'vectorstore' not in st.session_state:
        logger.info('Initializing vectorstore...')
        st.session_state.vectorstore = get_vectorstore(
            persist_directory=settings.VECTORSTORE_DIR,
            embeddings=st.session_state.embeddings
            )
        st.session_state.source_names = get_source_names(st.session_state.vectorstore)
        st.session_state.active_sources = []

    if 'llm' not in st.session_state:
        logger.info('Initializing LLM...')
        st.session_state.llm = YandexGPT(
            folder_id=settings.YANDEX_FOLDER_ID,
            oauth_token=settings.YANDEX_OAUTH_TOKEN
        )

    # Initialize sidebar
    st.sidebar.header("Выберите интересующие вас источники:")
    col1, col2 = st.sidebar.columns([1, 1])
    add_all_button = col1.button("Добавить все")
    clear_all_button = col2.button("Очистить все")

    if add_all_button:
        st.session_state.active_sources = st.session_state.source_names.copy()
    if clear_all_button:
        st.session_state.active_sources = []

    # Set selected sources
    selected_sources = []
    for source in st.session_state.source_names:
        checkbox_state = st.sidebar.checkbox(source,
                                             value=(source in st.session_state.active_sources),
                                             key=source)
        if checkbox_state:
            selected_sources.append(source)
    st.session_state.active_sources = selected_sources

    st.title("Text Mining Assistant")
    st.caption("Ассистент в области Text Mining")

    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "Добрый день! Я - ваш персональный ассистент в сфере Data Science и Text Mining. Как я могу помочь вам сегодня?", "docs": []}]

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg['content'])

    if prompt := st.chat_input():
        logger.info(f'Got message from user: {prompt}')
    
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)
        with st.spinner('Обработка запроса...'):    
            response = handle_user_input(question=prompt,
                                         llm=st.session_state.llm,
                                         vectorstore=st.session_state.vectorstore,
                                         sources=st.session_state.active_sources
            )
            logger.info(f'Got response from assistant based on {len(response["docs"])} docs: {response["llm_output"]}')

            msg = {
                "role": "assistant",
                "content": response["llm_output"],
                "docs": response["docs"]
            }
            st.session_state.messages.append(msg)

            with st.chat_message("assistant"):
                st.write(msg['content'])

                for i, doc in enumerate(msg["docs"]):
                    docname = doc.metadata.get("docname", "Unnamed Document")
                    # Create an expander for each document
                    with st.expander(f"Контекст {i+1}: {docname}", expanded=False):
                        st.write(doc.page_content)


if __name__ == '__main__':
    logger.info('Starting application...')
    main()
