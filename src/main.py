from enum import Enum
import streamlit as st
import extra_streamlit_components as stx
from typing import List
import logging
from datetime import datetime, timezone
import jwt

from stages import Stage, set_stage
from stages.auth import show_login_page, show_registratioin_page
from stages.chat import show_chat_page

_stages_view_mapper = {
    Stage.auth: show_login_page,
    Stage.registration: show_registratioin_page,
    Stage.chat: show_chat_page
}

from conf import settings

from logger import init_logs
init_logs()
logger = logging.getLogger(__name__)


@st.cache_resource
def get_manager():
    return stx.CookieManager()

def init_css():
    with open('static/css/styles.css') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

def validation_middleware(cookie_manager: stx.CookieManager) -> Stage:
    # Сначала проверяем токен в стейте сессии. Если его там нет - то валидируем в cookies.
    if (expiration_time := st.session_state.get('session_expires_at')) \
        and (expiration_time <= datetime.now(tz=timezone.utc)):
        return Stage.chat

    token = cookie_manager.get("token")
    if not token:
        return Stage.auth
    try:
        payload = jwt.decode(token, key=settings.JWT_SECRET_KEY, issuer=settings.JWT_ISSUER)
        st.session_state.session_expires_at = payload["exp"]
        st.session_state.user_id = payload["user_id"]
        return Stage.chat
    except jwt.ExpiredSignatureError:
        st.error('Истек срок действия сессии. Пожалуйста, пройдите авторизацию')
        return Stage.auth
    except jwt.exceptions.InvalidTokenError:
        return Stage.auth    


def main(): 
    st.set_page_config(page_title="Text Mining",
                       page_icon=":books:")
    cookie_manager = get_manager()

    stage = validation_middleware
    set_stage(stage)

    init_css()
    show_page_function = _stages_view_mapper[st.session_state.stage]
    show_page_function(cookie_manager)

if __name__ == '__main__':
    main()
