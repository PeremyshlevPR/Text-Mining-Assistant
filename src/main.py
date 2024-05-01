from enum import Enum
import streamlit as st
from typing import List
import logging
from datetime import datetime, timezone
import jwt
from streamlit_router import StreamlitRouter
from streamlit_cookies_manager import EncryptedCookieManager
from uuid import uuid4

from stages.auth import show_login_page, show_registratioin_page
from stages.chat import show_chat_page
from conf import settings

from logger import init_logs
init_logs()
logger = logging.getLogger(__name__)

class Router(StreamlitRouter):
    def redirect(self, path: str, method: str = None):
        self.reset_request_state()
        st.session_state['request'] = (path, method)
        st.session_state['request_id'] = uuid4().hex
        st.rerun()

    def serve(self):
        request = st.session_state.get('request')
        query_string = st.query_params
        if request:
            self.handle(*request)
            path, method = request
            query_string['request'] = [f'{method}:{path}']
            st.query_params = query_string
        elif 'request' in query_string:
            request = query_string.get('request')
            if isinstance(request, list):
                request = request[0]
            method, path = request.split(':')
            st.session_state['request'] = (path, method)
            st.rerun()
        else:
            self.handle(self.default_path)

def _init_css():
    with open('static/css/styles.css') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

def validate_session(router) -> None:
    """Функция валидирует то, что сессия актуальна и не просрочена.
    Если с сессией все в порядке - то возвращает None.
    Иначе - перенаправялет пользователя на страницу авторизации. 
    """
    # router = st.session_state.router

    if expiration_time := st.session_state.get('session_expires_at'):
        if datetime.now(tz=timezone.utc) <= expiration_time:
            logger.info(f'Got valid session from st.session_state for user {st.session_state.user_id}. Expires at {expiration_time}')
            return
    token = st.session_state.cookies.get("token")
    if not token:
        logger.info(f'Token not found in cookies, redirecting to authentification page.')
        router.redirect(*router.build("show_login_page"))
    try:
        payload = jwt.decode(token, algorithms="HS256", key=settings.JWT_SECRET_KEY, issuer=settings.JWT_ISSUER)
        st.session_state.session_expires_at = datetime.fromtimestamp(payload["exp"], tz=timezone.utc)
        st.session_state.user_id = payload["user_id"]
        return
    except jwt.ExpiredSignatureError:
        logger.info(f'Session expired, redirecting to authentification page.')
        st.error('Истек срок действия сессии. Пожалуйста, пройдите авторизацию')
        router.redirect(*router.build("show_login_page"))
    except jwt.exceptions.InvalidTokenError:
        logger.info(f'Could not validate session, redirecting to authentification page.')
        router.redirect(*router.build("show_login_page"))    

def chat_page_with_validation(router):
    validate_session(router)
    show_chat_page()

def _init_router() -> None:
    if "router" not in st.session_state:
        router = Router()
        router.register(chat_page_with_validation, '/')
        router.register(show_login_page, "/auth", methods=['POST'])
        router.register(show_registratioin_page, "/register", methods=['POST'])
        st.session_state.router = router

def _init_cookie_manager() -> None:
    if "cookies" not in st.session_state:
        cookies = EncryptedCookieManager(
            prefix="pperemyshlev/streamlit-cookies-manager/",
            password=settings.COOKIES_PASSWORD,
        )
        if not cookies.ready():
            st.stop()
        st.session_state.cookies = cookies

def main():
    st.set_page_config(page_title="Text Mining",
                       page_icon=":books:")
    
    st.session_state.authorized = False
    _init_cookie_manager()
    _init_router()
    _init_css()
    st.session_state.router.serve()

if __name__ == '__main__':
    main()
