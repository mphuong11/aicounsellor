import streamlit as st
import yaml
from passlib.hash import pbkdf2_sha256
import random
import string
from langchain.vectorstores import Chroma
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.llms import GooglePalm
from langchain.chains import RetrievalQA
from langchain import PromptTemplate
import google.generativeai as palm
from langchain.embeddings import GooglePalmEmbeddings
import torch
# Load configuration from YAML file
with open('config.yaml') as file:
    config = yaml.safe_load(file)

# Check if 'users' key exists in the configuration, if not, create an empty dictionary
if 'users' not in config:
    config['users'] = {}

# Initialize session state
if 'current_user' not in st.session_state:
    st.session_state.current_user = None

# Function to save the configuration to the YAML file
def save_config():
    with open('config.yaml', 'w') as file:
        yaml.dump(config, file, default_flow_style=False)
def home():
    st.subheader("Chatbot")
    # Set your Google Palm API key
    palm_api = "AIzaSyAWFYtQuHNIha4v2-vmDJPVopM-1exlMF0"

    # Initialize SentenceTransformer embeddings
    embeddings = SentenceTransformerEmbeddings(model_name="all-mpnet-base-v2")

    # Initialize Chroma for creating a database of chunks
    db = Chroma(persist_directory="./chroma_index_5", embedding_function=embeddings)

    # Set up Google Palm language model
    google_palm_llm = GooglePalm(google_api_key=palm_api)

    # Set up Streamlit app
    #st.title("College AI advisor")

    # User input
    user_input = st.text_input("Enter your context:")

    # User question

    # Button to trigger QA
    if st.button("Get Answer"):
        # Perform QA based on user input
        qa = RetrievalQA.from_chain_type(
            llm=google_palm_llm,
            chain_type="stuff",
            retriever=db.as_retriever(search_type="similarity", search_kwargs={"k": 8}),
            return_source_documents=True
        )
        
        # Set up question prompt
        question_template = """
        Utilize the provided context to respond to the question below.
        If you lack the information, kindly state that you don't know rather than providing speculative answers.
        Envision yourself advising a college student and communicate in a natural and friendly manner.
        Please use English exclusively; refrain from incorporating other languages.

        {context}
        Student Inquiry: {question}
        Advisor's Response:
        """
        QUESTION_PROMPT = PromptTemplate(
            template=question_template, input_variables=["question", "context"]
        )
        
        # Set up QA parameters
        qa.combine_documents_chain.llm_chain.prompt = QUESTION_PROMPT
        qa.combine_documents_chain.verbose = True
        qa.return_source_documents = True
        
        # Run QA
        result = qa({"query": user_input})
        
        # Display result
        st.write("Result:", result['result'])
def search_semantic():
    st.subheader("Search database")
    # Set your Google Palm API key
    palm_api = "AIzaSyAWFYtQuHNIha4v2-vmDJPVopM-1exlMF0"

    # Initialize SentenceTransformer embeddings
    embeddings = SentenceTransformerEmbeddings(model_name="all-mpnet-base-v2")

    # Initialize Chroma for creating a database of chunks
    db = Chroma(persist_directory="./chroma_index_5", embedding_function=embeddings)

    # Set up Google Palm language model
    google_palm_llm = GooglePalm(google_api_key=palm_api)

    # Set up Streamlit app
    #st.title("College AI advisor")

    # User input
    user_input = st.text_input("Enter your context:")

    # User question

    # Button to trigger QA
    if st.button("Get Answer"):
        # Perform QA based on user input
        qa = RetrievalQA.from_chain_type(
            llm=google_palm_llm,
            chain_type="stuff",
            retriever=db.as_retriever(search_type="similarity", search_kwargs={"k": 8}),
            return_source_documents=True
        )  
             
        # Run QA
        result = qa({"query": user_input})
        
        # Display result
        for re in result['source_documents']:
            st.write(re.page_content)

def main():
    st.title("College AI Advisor")

    menu = ["Login", "Home","Search", "Register", "Change Password", "Forgot Password", "Logout"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Home":
        if st.session_state.current_user:
            st.subheader("Welcome ",st.session_state.current_user)
            home()
        else:
            st.warning("Please login to use the function")   
    elif choice == "Login":
        if st.session_state.current_user:
            st.write("You has successfully signed in")
        else:
            login()
    elif choice == "Search":
        if st.session_state.current_user:
            search_semantic()
        #search_name()
        #search_semantic()
        else:
            st.warning("Please login to use the function") 
    elif choice == "Register":
        if st.session_state.current_user:
            st.warning("Please log out before registering")
        else:
            register()
    elif choice == "Change Password":
        if st.session_state.current_user:
            change_password()
        else:
            st.warning("Please login to change password.")
    elif choice == "Forgot Password":
        forgot_password()
    elif choice == "Logout":
        if st.session_state.current_user:
            logout()
        else:
            st.warning("You are not in any account")
       

def login():
    st.subheader("Login")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        st.session_state.current_user = login_user(username, password)

def register():
    st.subheader("Register")

    new_username = st.text_input("Username")
    new_password = st.text_input("Password", type="password")
    confirm_password = st.text_input("Confirm Password", type="password")

    if st.button("Register"):
        register_user(new_username, new_password)

def change_password():
    st.subheader("Change Password")

    current_password = st.text_input("Current Password", type="password")
    new_password = st.text_input("New Password", type="password")
    confirm_password = st.text_input("Confirm New Password", type="password")

    if st.button("Change Password"):
        change_password_user(st.session_state.current_user, current_password, new_password)

def forgot_password():
    st.subheader("Forgot Password")

    username = st.text_input("Username")

    if st.button("Reset Password"):
        reset_password(username)

def logout():
    st.session_state.current_user = None
    st.write("Your account has exited")

def login_user(username, password):
    if username in config['users']:
        stored_password = config['users'][username]['password']
        if pbkdf2_sha256.verify(password, stored_password):
            st.success("Logged in as {}".format(username))
            return username
        else:
            st.error("Incorrect password.")
    else:
        st.error("User does not exist.")

def register_user(username, password):
    if username not in config['users']:
        hashed_password = pbkdf2_sha256.hash(password)
        config['users'][username] = {'password': hashed_password}
        save_config()
        st.success("Registered successfully. You can now login.")
    else:
        st.warning("Username already exists. Choose a different username.")

def change_password_user(username, current_password, new_password):
    if username in config['users']:
        stored_password = config['users'][username]['password']
        if pbkdf2_sha256.verify(current_password, stored_password):
            hashed_new_password = pbkdf2_sha256.hash(new_password)
            config['users'][username]['password'] = hashed_new_password
            save_config()
            st.success("Password changed successfully.")
        else:
            st.error("Incorrect current password.")
    else:
        st.error("User does not exist.")

def reset_password(username):
    if username in config['users']:
        new_password = generate_random_password()
        hashed_new_password = pbkdf2_sha256.hash(new_password)
        config['users'][username]['password'] = hashed_new_password
        save_config()
        st.success("Password reset successfully. New password: {}".format(new_password))
    else:
        st.error("User does not exist.")

def generate_random_password(length=8):
    characters = string.ascii_letters + string.digits + string.punctuation
    return "".join(random.choice(characters) for i in range(length))


if __name__ == "__main__":
    main()
