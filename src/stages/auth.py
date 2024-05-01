import streamlit as st
import extra_streamlit_components as stx
from db import models
from db.crud import get_users, create_user
from sqlalchemy.exc import IntegrityError
from streamlit_router import StreamlitRouter
import utils
import re
import time

import logging
logger = logging.getLogger(__name__)


def authenticate(username: str, password: str) -> str:
    if user := get_users(return_one=True, username=username):
        hashed_password, _ = utils.hash_password(password, salt=user.password_salt)

        if hashed_password == user.password_hash:
            return utils.issue_token(user_id=user.id, role=user.role)
    return None
        

def show_login_page(router: StreamlitRouter):
    st.title("Авторизация")

    username = st.text_input(placeholder="Логин")
    password = st.text_input(placeholder="Пароль", type="password")

    login_button = st.button("Войти")

    if login_button:
        if token := authenticate(username, password):
            st.success("Успешная аутентификация!")
            st.session_state.token = token
            st.session_state.cookies["token"] = token
            st.session_state.cookies.save()

            logger.info(f'User {username} successfully authentificated')
            logger.info(f'Cookies: {st.session_state.cookies}')

            router.redirect(*router.build("chat_page_with_validation"))
        else:
            st.error("Ошибка аутентификации. Пожалуйста, проверьте учетные данные.")

    st.write("Еще нет аккаунта?")
    register_button = st.button("Регистрация")

    if register_button:
        if not password:

        router.redirect(*router.build("show_registratioin_page"))


def get_password_errors(password):
    errors = []
    if len(password) < 8:
        errors.append("Пароль должен содержать не менее 8 символов.")
    if not re.search(r"\d", password):
        errors.append("Пароль должен содержать хотя бы одну цифру.")
    return errors


def show_registratioin_page(router: StreamlitRouter):
    st.title("Регистрация")
    username = st.text_input("Введите логин")
    password = st.text_input("Введите пароль", type="password")
    confirm_password = st.text_input("Подтвердите пароль", type="password")
    
    col1, col2 = st.columns(2)
    return_button  = col1.button("Назад")
    register_button  = col2.button("Зарегистрироваться")

    if return_button:
        router.redirect(*router.build("show_login_page"))
    
    if register_button:
        if user := get_users(return_one=True, username=username):
            st.error(f'Пользователь с таким именем уже зарегистрирован, попробуйте другое')
        else:
            if not (errors := get_password_errors(password)):
                if password != confirm_password:
                    st.error("Пароли не совпадают")
                else:
                    user = create_user(username=username, password=password, role="user")
                    token = utils.issue_token(user_id=user.id, role=user.role)

                    st.session_state.token = token
                    st.session_state.cookies["token"] = token
                    router.redirect(*router.build("chat_page_with_validation"))
            else:
                st.error("\n".join(errors))
