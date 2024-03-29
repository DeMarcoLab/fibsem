import streamlit as st

from fibsem import config as cfg
from fibsem.db.v3.util import get_session, create_connection

import pandas as pd
st.set_page_config(page_title='Fibsem DB v3', page_icon=":microscope:", layout="wide")

st.title('Fibsem DB v3')


# connectto the database
conn = st.connection("fibsem.db", type="sql", url=f'sqlite:///{cfg.DATABASE_PATH}', ttl=0)



@st.cache_data(show_spinner=True)
def get_user_data():
    return conn.query("SELECT * FROM user")


# force refresh
if st.button('Refresh'):
    get_user_data.clear()

df = None

df = get_user_data()
st.data_editor(df)









# sidebar
st.sidebar.title('Navigation')
page = st.sidebar.radio('Go to', ('Home', 'Users', 'Samples', 'Instruments', 'Configurations'))

@st.cache_resource
def get_user_model():
    from fibsem.db.v3.models import User
    return User

User = get_user_model()

# create a form for adding users (using sqlmodel)
if page == 'Users':
    st.header('Users')
    with st.form(key='add_user'):
        username = st.text_input('Username')
        name = st.text_input('Name')
        email = st.text_input('Email')
        password = st.text_input('Password', type='password')
        role = st.selectbox('Role', ['admin', 'user'])
        submit = st.form_submit_button('Add User')


        if submit:
            user = User(username=username, name=name, email=email, password=password, 
                        role=role)
            engine = create_connection(echo=True)
            with get_session(engine) as session:
                session.add(user)
                session.commit()
            st.toast(f'User {username} added successfully')

            # rerun
            st.rerun()

    # add a form to delete users
    with st.form(key='delete_user'):
        user_id = st.text_input('User ID')
        submit = st.form_submit_button('Delete User')

        if submit:
            engine = create_connection(echo=True)
            with get_session(engine) as session:
                user = session.get(User, user_id)
                session.delete(user)
                session.commit()
            st.toast(f'User {user_id} deleted successfully')

            # rerun
            st.rerun()
