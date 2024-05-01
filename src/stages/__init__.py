from enum import Enum
import streamlit as st

class Stage(Enum):
    auth = 'auth'
    registration = 'registration'
    chat = 'chat'


def set_stage(stage: Stage):
    # Меняем стадию только, если в текущий момент времени находимся на другой стадии
    if current_stage := st.session_state.get("stage"):
        if (current_stage == stage):
            return
    st.session_state["stage"] = stage
    st.rerun()
