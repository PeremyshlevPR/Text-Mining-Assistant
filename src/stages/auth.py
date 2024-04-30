import streamlit as st
import extra_streamlit_components as stx
from db import models
from db.crud import get_users, create_user
from sqlalchemy.exc import IntegrityError

import utils
from . import set_stage, Stage

def authenticate(username: str, password: str) -> str:
    if user := get_users(return_one=True, username=username):
        hashed_password, _ = utils.hash_password(password, salt=user.password_salt)

        if hashed_password == user.password_hash:
            return utils.issue_token(user_id=user.id, role=user.role)
    return None
        

def register(username: str, password: str) -> models.User:
    try:
        return create_user(username=username, password=password, role="user")
    except IntegrityError:
        return None

def show_login_page(cookie_manager: stx.CookieManager):
    st.title("Авторизация")

    username = st.text_input("Логин")
    password = st.text_input("Пароль", type="password")

    login_button = st.button("Войти")

    if login_button:
        if token := authenticate(username, password):
            st.success("Успешная аутентификация!")
            st.session_state.token = token
            cookie_manager.set("token", token)
            set_stage(Stage.chat)
        else:
            st.error("Ошибка аутентификации. Пожалуйста, проверьте учетные данные.")

    st.write("Еще нет аккаунта?")
    register_button = st.button("Регистрация")

    if register_button:
        set_stage(Stage.registration)


def show_registratioin_page(cookie_manager: stx.CookieManager):
    st.title("Регистрация")
    username = st.text_input("Введите логин")
    password = st.text_input("Введите пароль", type="password")
    confirm_password = st.text_input("Подтвердите пароль", type="password")
    
    col1, col2 = st.columns(2)
    return_button  = col1.button("Назад")
    register_button  = col2.button("Зарегистрироваться")

    if return_button:
        set_stage(Stage.auth)
    
    if register_button:
        if password == confirm_password:
            if user := register(username, password):
                token = utils.issue_token(user_id=user.id, role=user.role)

                st.session_state.token = token
                cookie_manager.set("token", token)
                set_stage(Stage.chat)
            else:
                st.error("Ошибка регистрации. Попробуйте снова.")
        else:
            st.error("Пароли не совпадают.")
