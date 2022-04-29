import time
import streamlit as st

# How to use:
# ============
# msg = string
# wait = integer - default to 3
# type_ = string - default to warning (success, warning, danger)

def customMsg(msg, wait=3, type_='warning'):
    placeholder = st.empty()
    styledMsg = f'\
        <div class="element-container" style="width: 693px;">\
            <div class="alert alert-{type_} stAlert" style="width: 693px;">\
                <div class="markdown-text-container">\
                    <p>{msg}</p></div></div></div>\
    '
    placeholder.markdown(styledMsg, unsafe_allow_html=True)
    time.sleep(wait)
    placeholder.empty()


def customMsg2(msg, wait=3, type_='warning'):
    placeholder = st.empty()
    if type_=="success":
        styledMsg="ACCESS IS ALLOWED, Please approach hand sanitizer"
        placeholder.success(styledMsg)
    else:
        styledMsg="ACCESS DENIED"
        placeholder.warning(styledMsg)

    time.sleep(wait)
    placeholder.empty()
 

    