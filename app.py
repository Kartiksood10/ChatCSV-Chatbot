import streamlit as st
from streamlit_chat import message
# Python tempfile module allows you to create a temporary file and perform various operations on it
import tempfile 
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import CTransformers
from langchain.chains import ConversationalRetrievalChain

# path to store embeddings in a vectorDB
DB_FAISS_PATH = 'vectorstore/db_faiss'

#Loading the model
def load_llm():
    # Load the locally downloaded model here
    llm = CTransformers(
        model = "llama-2-7b-chat.ggmlv3.q8_0.bin",
        model_type="llama",
        max_new_tokens = 512,
        temperature = 0
    )
    return llm

st.header("Chat with CSV using llama2🦙🦜")
st.markdown("Talk to your CSV Files with Chat CSV Chatbot❤️!")
# st.markdown("<h3 style=color: white;'>Built by <a href='https://github.com/Kartiksood10'>Kartik Sood with❤️</a></h3>", unsafe_allow_html=True)

uploaded_file = st.sidebar.file_uploader("Upload your CSV file:", type="csv")

if uploaded_file:
    # The NamedTemporaryFile() function creates a file with a visible name in the file system. It takes a delete parameter which we can set as False to prevent the file from being deleted when it is closed.
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        # The write() method is used to write to a temporary file
        # .getvalue() accesses the content present in the csv file
        tmp_file.write(uploaded_file.getvalue())
        # .name returns the csv file path
        tmp_file_path = tmp_file.name
        print(tmp_file_path)
        #print(uploaded_file.getvalue())

    loader = CSVLoader(file_path=tmp_file_path, encoding="utf-8", csv_args={'delimiter': ','})
    data = loader.load()
    #print(data)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={'device': 'cpu'})

    db = FAISS.from_documents(data, embeddings)
    db.save_local(DB_FAISS_PATH)
    llm = load_llm()

    chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=db.as_retriever())

    # function that return the response from the chain
    def conversational_chat(query):
        result = chain({'question': query, "chat_history": st.session_state['history']})
        print(result)
        st.session_state['history'].append((query, result["answer"]))
        return result['answer']
    
    # session state that stores chat history
    if 'history' not in st.session_state:
        st.session_state['history'] = []

    # session state that stores generated response by LLM
    if 'generated' not in st.session_state:
        st.session_state['generated'] = ["Hello, Ask me anything about " + uploaded_file.name + "🤗"]

    # session state that stores the input queries of the user
    if 'past' not in st.session_state:
        st.session_state['past'] = ["Hey ! 👋"]

    # container for the chat history

    # .container() Inserts an invisible container into your app that can be used to hold multiple elements. This allows you to, for example, insert multiple elements into your app out of order.

    # response_container = st.container()
    # container = st.container()

    # A form is a container that visually groups other elements and widgets together, and contains a Submit button. When the form's Submit button is pressed, all widget values inside the form will be sent to Streamlit in a batch.
    # with container:
    #     # clear_on_submit - all values return to default value after submit
    #     with st.form(key="my_form", clear_on_submit=True):
    #         user_input = st.text_input("Query:", placeholder="Chat with your CSV data here...", key='input')
    #         submit_button = st.form_submit_button(label="Submit")
    
    user_input = st.chat_input("Chat with your CSV file...")

    if user_input:
        with st.spinner("Generating response..."):

            output = conversational_chat(user_input)
            st.session_state['past'].append(user_input)
            st.session_state['generated'].append(output)

    if st.session_state['generated']:
            for i in range(len(st.session_state['generated'])):
                # is_user=True aligns user messages to the right as in a chatbot interface via streamlit-chat
                message(st.session_state['past'][i], is_user=True, key=str(i) + '_user', avatar_style="big-smile")
                message(st.session_state["generated"][i], key=str(i), avatar_style="thumbs")