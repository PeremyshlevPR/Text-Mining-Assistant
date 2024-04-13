import streamlit as st
from typing import List
import logging

from chromadb import Embeddings
from langchain_community.vectorstores import Chroma
from embeddings import TogetherEmbeddings
from translators import YandexTranslator
from llms import YandexGPT
from conf.settings import settings
from prompt_templates import QUERY_WITH_CONTEXT_TEMPLATE

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

    books = set()
    articles = set()
    for meta in result['metadatas']:
        if meta['source_type'] == "book":
            books.add(meta["docname"])
        else:
            articles.add(meta["docname"])

    books = sorted(list(books))
    articles = sorted(list(articles))
    logger.info(f'Vectorstore has documents from {len(books)} books and {len(articles)} articles.')

    return books, articles

def get_relevant_docs(vectorstore, query, sources, k=3):
    filter_ = {
        'docname': {
            '$in': sources
        }
    }
    response = vectorstore.similarity_search_with_score(query, filter=filter_, k=k)
    return [doc for doc, score in response]


def handle_user_input(messages, llm, vectorstore, sources, n_docs=3):
    if sources:
        logger.info(f'Soures passed: {", ".join(sources)}')

        query = "\n".join([msg["en_text"] for msg in messages[-3:]])
        print(query)
        docs = get_relevant_docs(vectorstore, query=query, sources=sources, k=n_docs)
        
        for i, doc in enumerate(docs):
            docs[i].page_content = _translator.translate(doc.page_content, target='ru')[0]
        context = '\n\n'.join([doc.page_content for doc in docs])

        logger.info(f'Documents succesfully retrieved from sources: {", ".join([doc.metadata["docname"] for doc in docs])}')

        prompt = QUERY_WITH_CONTEXT_TEMPLATE.format(question=messages[-1]["text"], context=context)
    else:
        logger.info('')
        docs = []
        prompt = messages[-1]["text"]

    messages_to_llm = messages[:-1] + [{"role": "user", "text": prompt}]
    response = llm(messages=messages_to_llm)
    return {
        'llm_output': response,
        'en_llm_output': _translator.translate(response, target='en')[0],
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
        st.session_state.book_names, st.session_state.article_names = get_source_names(st.session_state.vectorstore)
        st.session_state.active_books = []
        st.session_state.active_articles = []

    if 'llm' not in st.session_state:
        logger.info('Initializing LLM...')
        st.session_state.llm = YandexGPT(
            folder_id=settings.YANDEX_FOLDER_ID,
            oauth_token=settings.YANDEX_OAUTH_TOKEN
        )

    # Initialize sidebar
    st.sidebar.header("Выберите интересующие вас источники")

    # Books
    st.sidebar.subheader("Книги")
    books_col1, books_col2 = st.sidebar.columns([1, 1])
    books_add_all_button = books_col1.button("Добавить все", key='books_add')
    books_clear_all_button = books_col2.button("Очистить все", key='books_del')

    if books_add_all_button:
        st.session_state.active_books = st.session_state.book_names.copy()
    if books_clear_all_button:
        st.session_state.active_books = []

    # Set selected books
    selected_books = []
    for book in st.session_state.book_names:
        checkbox_state = st.sidebar.checkbox(book,
                                             value=(book in st.session_state.active_books),
                                             key=book)
        if checkbox_state:
            selected_books.append(book)
    st.session_state.active_books = selected_books

    for _ in range(2):
        st.sidebar.write("")
    
    # Articles
    st.sidebar.subheader("Научные статьи")
    articles_col1, articles_col2 = st.sidebar.columns([1, 1])
    articles_add_all_button = articles_col1.button("Добавить все", key='articles_add')
    articles_clear_all_button = articles_col2.button("Очистить все", key='articles_del')

    if articles_add_all_button:
        st.session_state.active_articles = st.session_state.article_names.copy()
    if articles_clear_all_button:
        st.session_state.active_articles = []

    # Set selected articles
    selected_articles = []
    for article in st.session_state.article_names:
        checkbox_state = st.sidebar.checkbox(article,
                                             value=(article in st.session_state.active_articles),
                                             key=article)
        if checkbox_state:
            selected_articles.append(article)
    st.session_state.active_articles = selected_articles
    

    st.title("Text Mining Assistant")
    st.caption("Ассистент в области Text Mining")

    if "messages" not in st.session_state:
        system_message = {
            "role": "assistant",
            "text": "Добрый день! Я - ваш персональный ассистент в сфере Data Science и Text Mining. Как я могу помочь вам сегодня?",
            "en_text": "Hello! I am your personal assistant in the field of Data Science and Text Mining. How can I help you today?",
            "docs": []
        }
        st.session_state.messages = [system_message]

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg['text'])

    if prompt := st.chat_input():
        logger.info(f'Got message from user: {prompt}')
    
        with st.chat_message("user"):
            st.write(prompt)

        with st.spinner('Обработка запроса...'):
            en_prompt = _translator.translate(prompt, target='en')[0]
            st.session_state.messages.append({"role": "user", "text": prompt, "en_text": en_prompt})

            response = handle_user_input(
                messages=st.session_state.messages[-5:],
                llm=st.session_state.llm,
                vectorstore=st.session_state.vectorstore,
                sources=st.session_state.active_books + st.session_state.active_articles
            )
            logger.info(f'Got response from assistant based on {len(response["docs"])} docs: {response["llm_output"]}')

            msg = {
                "role": "assistant",
                "text": response["llm_output"],
                "en_text": response["en_llm_output"],
                "docs": response["docs"]
            }
            st.session_state.messages.append(msg)

        with st.spinner(False):  # Turn off the spinner after processing
            with st.chat_message("assistant"):
                st.write(msg['text'])
    
                for i, doc in enumerate(msg["docs"]):
                    docname = doc.metadata.get("docname", "Unnamed Document")
    
                    with st.expander(f"Контекст {i+1}: {docname}", expanded=False):
                        st.write(doc.page_content)


if __name__ == '__main__':
    logger.info('Starting application...')
    main()
