import time
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import PromptTemplate
from operator import itemgetter
import logging
import pandas as pd
import os
from tqdm import tqdm
import pickle
from PIL import Image
import numpy as np
import platform
import subprocess
from langchain.schema import Document
import tempfile
from streamlit_extras.switch_page_button import switch_page
from concurrent.futures import ThreadPoolExecutor

# Initialize embeddings


img_embeddings = OllamaEmbeddings(model='bge-m3')
st.sidebar.header("Model Selection 🧠")
model_options = {
    "Llama 3.1": "llama3.1",
    "Codellama": "codellama",
    "llava-llama3(Muti-model)" : "llava-llama3",
    "Mistral":"mistral",
    "qwen2.5-coder" : "qwen2.5-coder"
}
selected_model = st.sidebar.selectbox("Select Language Model", options=list(model_options.keys()))
model_name = model_options[selected_model]


# Initialize logging
logging.basicConfig(level=logging.INFO)
start_time = time.time()



if 'username' not in st.session_state:
    st.session_state['username'] = ""
    

username_input = st.text_input("Enter your username to start:", value=st.session_state['username'])

if username_input:
    st.session_state['username'] = username_input
else:
    st.stop()  

# Initialize Streamlit application
 



 # Log the time after entering the username
logging.info(f"User logged in as: {st.session_state['username']}")
login_time = time.time()
logging.info(f"Time after login: {login_time - start_time:.2f} seconds")

st.title(f"🤖 Bodhee Bot | Welcome, {st.session_state['username']}!")

def save_chat_history_with_feedback():
    if st.session_state.get("messages"):
        # Initialize chat history string with the username
        chat_history = f"Chat History for {st.session_state['username']}:\n\n"
        
        # Append each message with proper role formatting (User or Bot) and feedback
        for i, msg in enumerate(st.session_state['messages']):
            if msg['role'].lower() == "user":
                chat_history += f"{st.session_state['username']}: {msg['content']}\n"
            else:
                chat_history += f"Bot: {msg['content']}\n"
                # Append feedback if available
                if 'feedback' in msg:
                    feedback = 'Positive' if msg['feedback'] else 'Negative'
                    chat_history += f"Feedback: {feedback}\n"
        
        # Define the file path with username included
        file_path = f"{st.session_state['username']}_chat_history_with_feedback.txt"
        
        # Write the chat history to the file
        with open(file_path, "w") as f:
            f.write(chat_history)
        
        return file_path
    return None

def save_chat_history():
    if st.session_state.get("messages"):
        # Initialize chat history string with the username
        chat_history = f"Chat History for {st.session_state['username']}:\n\n"
        
        # Append each message with proper role formatting (User or Bot)
        for msg in st.session_state['messages']:
            if msg['role'].lower() == "user":
                chat_history += f"{st.session_state['username']}: {msg['content']}\n"
            else:
                chat_history += f"Bot: {msg['content']}\n"
        
        # Define the file path with username included
        file_path = f"{st.session_state['username']}_chat_history.txt"
        
        # Write the chat history to the file
        with open(file_path, "w") as f:
            f.write(chat_history)
        
        return file_path
    return None


# Define prompt templates
prompt_templates = {
    "detailed": """
    You are an assistant that provides comprehensive, detailed answers to questions based on
    a given context. Provide in-depth explanations and cover all aspects of the topic. If user asks for image give response only for query.

    Context: {context}

    Question: {question}
    """,
    
    "concise": """
    You are an assistant that provides concise, to-the-point answers to questions based on
    a given context. Keep your responses short and clear. If user asks for image just ignore that and give response only for query.

    Context: {context}

    Question: {question}
    """
}

# Keywords that trigger the 'detailed' prompt template
detailed_keywords = ["explain", "detail", "describe", "in-depth", "thorough", "elaborate"]

image_terms = ["picture", "photo", "image", "graphic", "image?", "artwork", "illustration", 
               "snapshot", "scenery", "vista", "view", "scene", "photograph", "painting", 
               "drawing", "sketch", "portrait", "landscape", "cityscape", "seascape", 
               "still life", "abstract art", "digital art", "cartoon", "anime", 
               "infographic", "diagram", "chart", "graph"]

# Function to choose the appropriate prompt template
def select_prompt_template(question):
    if any(keyword in question.lower() for keyword in detailed_keywords):
        return prompt_templates["detailed"]
    else:
        return prompt_templates["concise"]

vectorstore = None
file_embeddings = OllamaEmbeddings(model='mxbai-embed-large')

uploaded_file = st.file_uploader("Upload a file (PDF, Excel, or CSV)", type=["pdf", "xlsx", "csv"])

if uploaded_file is not None:
    # Get the name of the uploaded file without the extension
    file_name = os.path.splitext(uploaded_file.name)[0]

    # Path for saving the vector store
    VECTORSTORE_PATH = os.path.join("/home/localstudio/rag-demo/rag_chatbot/upload_vector", f"{file_name}_vectorstore.pkl")

    # Check if vector store already exists
    if os.path.exists(VECTORSTORE_PATH):
        st.info("Loading existing vectorstore from disk...")
        # Load the vector store from the file
        with open(VECTORSTORE_PATH, 'rb') as f:
            vectorstore = pickle.load(f)
    else:
        st.info("No existing vectorstore found. Creating a new one...")

        # Handle file types and create embeddings if vector store doesn't exist
        if uploaded_file.name.endswith('.pdf'):
            # Handle PDF
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
                temp_file.write(uploaded_file.read())
                temp_pdf_path = temp_file.name
            loader = PyPDFLoader(temp_pdf_path)
            pages = loader.load()

            splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
            chunks = splitter.split_documents(pages)

        # elif uploaded_file.name.endswith('.xlsx'):
        #     # Handle Excel files
        #     df = pd.read_excel(uploaded_file)
        #     documents = [Document(page_content=row.to_string()) for _, row in df.iterrows()]

        #     # Optionally split documents
        #     splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        #     chunks = splitter.split_documents(documents)


        elif uploaded_file.name.endswith('.xlsx'):
        # Handle Excel files
            df = pd.read_excel(uploaded_file)

            # Combine rows into larger chunks for vectorization
            documents = [Document(page_content='\n'.join(df.iloc[i:i+10].to_string() for i in range(0, len(df), 10)))]

            # Adjust chunk size and overlap to reduce unnecessary chunking
            splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            chunks = splitter.split_documents(documents)

            # Rest of the code remains the same



        elif uploaded_file.name.endswith('.csv'):
            # Handle CSV files
            df = pd.read_csv(uploaded_file)
            documents = [Document(page_content=row.to_string()) for _, row in df.iterrows()]

            # Optionally split documents
            splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
            chunks = splitter.split_documents(documents)

        # Function to process a batch of chunks
        def process_chunks(batch):
            return file_embeddings.embed_documents([chunk.page_content for chunk in batch])

        start_time_file = time.time()

        # Create embeddings and vectorstore using parallel processing
        with st.spinner("Creating embeddings and vectorstore..."):
            # Using batch size to process multiple chunks at once
            batch_size = 10  # Adjust the batch size as necessary
            embeddings_list = []

            # Add a progress bar using tqdm
            with tqdm(total=len(chunks), desc="Vectorizing chunks", unit="chunk") as pbar:
                with ThreadPoolExecutor() as executor:
                    for i in range(0, len(chunks), batch_size):
                        batch = chunks[i:i + batch_size]
                        # Submit batch processing
                        embeddings = executor.submit(process_chunks, batch)
                        embeddings_list.extend(embeddings.result())  # Collect results
                        pbar.update(len(batch))  # Update progress bar for each batch processed

            # Create vectorstore with the processed embeddings
            vectorstore = FAISS.from_documents(chunks, file_embeddings)

            # Save vectorstore to disk
            with open(VECTORSTORE_PATH, 'wb') as f:
                pickle.dump(vectorstore, f)

        end_time_file = time.time()
        time_taken = end_time_file - start_time_file
        st.success(f"Embeddings created successfully in {time_taken:.2f} seconds!")

# once we upload can  we store this and then query  thus makiing this faster so that it will realoda eveytimeS

    # # Create embeddings and vectorstore
    # logging.info("Creating x` and vectorstore...")
    # vectorstore = FAISS.from_documents(chunks, img_embeddings)
    # logging.info("Embeddings and vectorstore created.")
else:



# Define the PDFs and vectorstore paths
    pdf_options = {
        "BPS": '/home/localstudio/rag-demo/local-rag-example/DES1005-BPS_Scheduling_Engine_Design_Document_v1.0.pdf',
        "CPA": '/home/localstudio/rag-demo/local-rag-example/Change Point Analyzer -Training Manual 1.pdf',
        "Material": '/home/localstudio/rag-demo/local-rag-example/Material Constraint Training Manual (1).pdf',
        "BPS-Equipment Constraint" : "/home/localstudio/rag-demo/rag_chatbot/new_docs/L8-002-1-Neewee Bodhee Production Scheduler -Planner Constraints Training Module - Equipment.pdf",
        "BPS-Employee Constraints" : "/home/localstudio/rag-demo/rag_chatbot/new_docs/L8-002-2-Neewee Bodhee Production Scheduler -Planner Constraints Training Module - Employee.pdf",
        "BPS-Material Constraints" : "/home/localstudio/rag-demo/rag_chatbot/new_docs/L8-002-3-Neewee Bodhee Production Scheduler -Planner Constraints Training Module - Material.pdf",
        "BPS-OptimizationObjectives & Assumption" : "/home/localstudio/rag-demo/rag_chatbot/new_docs/L8-002-4-Neewee Bodhee Production Scheduler -Planner -Objective and Optimization Training Manual.pdf",
        "BPS-Micro Planner/Scheduler": "/home/localstudio/rag-demo/rag_chatbot/new_docs/L8-017- Neewee Bodhee Production Scheduler - Micro Planner Training Manual.pdf",
        "BPS-Change Point Analyzer": "/home/localstudio/rag-demo/rag_chatbot/new_docs/L8-018- Neewee Bodhee Production Scheduler - Change Point Analyzer.pdf"
    }

    vectorstore_paths = {
        "BPS": '/home/localstudio/rag-demo/rag_chatbot/embeddings/bps.pkl',
        "CPA": '/home/localstudio/rag-demo/rag_chatbot/embeddings/cpa.pkl',
        "Material": '/home/localstudio/rag-demo/rag_chatbot/embeddings/materail.pkl',
        "BPS-Equipment Constraint": '/home/localstudio/rag-demo/rag_chatbot/embeddings/BPS-Equipment_Constraint.pkl',
        "BPS-Employee Constraints" : '/home/localstudio/rag-demo/rag_chatbot/embeddings/BPS-Employee_Constraints.pkl',
        "BPS-Material Constraints" : '/home/localstudio/rag-demo/rag_chatbot/embeddings/BPS-Material_Constraints.pkl',
        "BPS-OptimizationObjectives & Assumption" : '/home/localstudio/rag-demo/rag_chatbot/embeddings/BPS-OptimizationObjectives_Assumption.pkl',
        "BPS-Micro Planner/Scheduler" : '/home/localstudio/rag-demo/rag_chatbot/embeddings/bps_micro_planner.pkl',
        "BPS-Change Point Analyzer" : '/home/localstudio/rag-demo/rag_chatbot/embeddings/BPS-Change_Point_Analyzer.pkl',

    }


# PDF file selection widget






    selected_pdf = st.selectbox("Select a PDF", options=list(pdf_options.keys()))

    PDF_FILE = pdf_options[selected_pdf]
    VECTORSTORE_PATH = vectorstore_paths[selected_pdf]


    if os.path.exists(VECTORSTORE_PATH):
        logging.info("Loading existing vectorstore...")
        with open(VECTORSTORE_PATH, 'rb') as f:
            vectorstore = pickle.load(f)
        logging.info("Vectorstore loaded from disk.")
    else:
        # Load the PDF if no vectorstore is found
        logging.info("Loading the PDF...")
        loader = PyPDFLoader(PDF_FILE)
        pages = loader.load()
        logging.info("PDF loaded.")

        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        chunks = splitter.split_documents(pages)
        vectorstore = FAISS.from_documents(chunks, img_embeddings)

        # Save the vectorstore to disk
        logging.info("Saving vectorstore to disk...")
        with open(VECTORSTORE_PATH, 'wb') as f:
            pickle.dump(vectorstore, f)
        logging.info("Vectorstore saved to disk.")
    


# # Image retrieval file paths
# csv_file_path = '/home/localstudio/rag-demo/rag_chatbot/cpa_dumps/CPA_description.csv'
# faiss_index_path = '/home/localstudio/rag-demo/rag_chatbot/cpa_dumps/CPA_image_faiss_index.pkl'
# faiss_data_path = '/home/localstudio/rag-demo/rag_chatbot/cpa_dumps/CPA_faiss_data.pkl'
# mapping_file_path = '/home/localstudio/rag-demo/rag_chatbot/cpa_dumps/CPA_image_summary.csv'




# Image retrieval file paths
csv_file_path = '/home/localstudio/rag-demo/rag_chatbot/matdumps/cpa_bps_scheduling description.csv'
faiss_index_path = '/home/localstudio/rag-demo/rag_chatbot/matdumps/cpa_bps_scheduling_image_faiss_index.pkl'
faiss_data_path = '/home/localstudio/rag-demo/rag_chatbot/matdumps/cpa_bps_scheduling_faiss_data.pkl'
mapping_file_path = '/home/localstudio/rag-demo/rag_chatbot/matdumps/cpa_bps_scheduling_image_summary.csv'

def open_file(path):
    if platform.system() == "Windows":
        os.startfile(path)
    elif platform.system() == "Darwin":  # macOS
        subprocess.call(["open", path])
    elif platform.system() == "Linux":
        try:
            subprocess.call(["xdg-open", path])
        except FileNotFoundError:
            try:
                subprocess.call(["gio", "open", path])
            except FileNotFoundError:
                raise NotImplementedError("No suitable command found to open files on this Linux system")
    else:
        raise NotImplementedError("Unsupported operating system")

# Load CSV file and verify contents
try:
    df = pd.read_csv(csv_file_path)
except FileNotFoundError:
    st.error(f"File not found: {csv_file_path}")
    raise

# Check columns in CSV
required_columns = ['image_file', 'description']
for col in required_columns:
    if col not in df.columns:
        st.error(f"Missing required column: {col}")
        raise KeyError(f"Missing required column: {col}")

# Check if FAISS index and mapping file already exist
if os.path.exists(faiss_index_path) and os.path.exists(faiss_data_path) and os.path.exists(mapping_file_path):
    logging.info("Loading existing FAISS store...")
    with open(faiss_index_path, 'rb') as f:
        index = pickle.load(f)

    with open(faiss_data_path, 'rb') as f:
        docstore, index_to_docstore_id = pickle.load(f)

    img_vectorstore = FAISS(index=index, docstore=docstore, index_to_docstore_id=index_to_docstore_id, embedding_function=img_embeddings)

    mapping_df = pd.read_csv(mapping_file_path)

else:
    st.write("FAISS index not found. Creating a new FAISS store.")
    documents = [Document(page_content=row['description'], metadata={'faiss_id': i, 'image_file': row['image_file']})
                 for i, row in df.iterrows()]

    img_vectorstore = FAISS.from_documents(documents, img_embeddings)

    image_paths = df['image_file'].tolist()
    image_path_mapping = {i: image_paths[i] for i in range(len(image_paths))}

    with open(faiss_index_path, 'wb') as f:
        pickle.dump(img_vectorstore.index, f)

    with open(faiss_data_path, 'wb') as f:
        pickle.dump((img_vectorstore.docstore, img_vectorstore.index_to_docstore_id), f)

    mapping_df = pd.DataFrame(list(image_path_mapping.items()), columns=['faiss_id', 'image_path'])
    mapping_df.to_csv(mapping_file_path, index=False)

# Load the vectorstore
if VECTORSTORE_PATH is not None:
    if os.path.exists(VECTORSTORE_PATH):
        logging.info("Loading existing vectorstore...")
        with open(VECTORSTORE_PATH, 'rb') as f:
            vectorstore = pickle.load(f)
        logging.info("Vectorstore loaded from disk.")
    else:
        logging.info("Loading the PDF...")
        loader_start = time.time()
        loader = PyPDFLoader(temp_pdf_path)
        pages = loader.load()
        loader_end = time.time()
        logging.info(f"PDF loaded in {loader_end - loader_start:.2f} seconds")

        logging.info("Splitting the document into chunks...")
        split_start = time.time()

        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        chunks = splitter.split_documents(pages)

        split_end = time.time()
        logging.info(f"Document split into chunks in {split_end - split_start:.2f} seconds")

        logging.info("Creating embeddings and vectorstore...")
        embed_start = time.time()

        embeddings = OllamaEmbeddings(model='mxbai-embed-large')
        vectorstore = FAISS.from_documents(chunks, embeddings)
        embed_end = time.time()
        logging.info(f"Embeddings and vectorstore created in {embed_end - embed_start:.2f} seconds")

        logging.info("Saving vectorstore to disk...")
        with open(VECTORSTORE_PATH, 'wb') as f:
            pickle.dump(vectorstore, f)
        logging.info("Vectorstore saved to disk.")

    logging.info("Initializing retriever and model...")
    retriever = vectorstore.as_retriever()
    
    #model = ChatOllama(model="llama3.1", temperature=0, streaming=True)
    model = ChatOllama(model=model_name, temperature=0, streaming=True)
    parser = StrOutputParser()

    # Initialize session state for chat history
    # Initialize session state for chat history
# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "Bot", "content": "Hello! How can I assist today?"}]
    st.session_state["first_prompt"] = None  # To store the first prompt
    st.session_state["chat_name"] = "Chat-Default"  # Initialize chat_name
    

# Render chat history
for idx, message in enumerate(st.session_state["messages"]):
    if message["role"] == "user":
        st.chat_message("user").write(f"👤 {message['content']}")
    else:
        st.chat_message("bot").write(f"🤖 {message['content']}")
        # Add thumbs up and thumbs down buttons for feedback
        feedback_col = st.columns(2)
        with feedback_col[0]:
            if st.button("👍", key=f"thumbs_up_{idx}"):  # Use index for unique key
                st.session_state.messages[idx]['feedback'] = True  # Add feedback
                
                #st.markdown("<p style='color:green;'>Thank you for your feedback! We're glad you liked it!</p>", unsafe_allow_html=True)
        with feedback_col[1]:
            if st.button("👎", key=f"thumbs_down_{idx}"):  # Use index for unique key
                st.session_state.messages[idx]['feedback'] = False  # Add feedback
                #st.success("")
                st.markdown("<p style='color:red;'>Thank you for your feedback! We're sorry to hear that. Please let us know how we can improve.</p>", unsafe_allow_html=True)
                #st.success("Thank you for your feedback! We're sorry to hear that. Please let us know how we can improve.")

# User input for questions
if question := st.chat_input("Type your question here..."):
    st.session_state.messages.append({"role": "user", "content": question})
    st.session_state.show_image = False  
    # Store the first prompt for the chat name
    if st.session_state["first_prompt"] is None:
        st.session_state["first_prompt"] = question[:5]  # Get the first 5 letters
        st.session_state["chat_name"] = f"Chat-{st.session_state['first_prompt']}"  # Create chat name

    # Display the chat name
    st.write(f"**Chat Name:** {st.session_state['chat_name']}")

    logging.info("Retrieving context from vectorstore...")
    retrieve_start = time.time()
    context = '\n'.join(map(lambda doc: doc.page_content, vectorstore.similarity_search(question, k=3)))
    retrieve_end = time.time()
    logging.info(f"Context retrieved in {retrieve_end - retrieve_start:.2f} seconds")

    selected_prompt_template = select_prompt_template(question)

    prompt = PromptTemplate.from_template(selected_prompt_template)
    chain = (
        {
            "context": itemgetter("question") | retriever,
            "question": itemgetter("question"),
        }
        | prompt
        | model
        | parser
    )

    # Display user question
    st.chat_message("user").write(question)
    #st.chat_message("user").write(f"👤 {question}")  
    chain_result = {"context": context, "question": question}  # Pass context and question

    logging.info("Generating answer from the model...")
    model_start = time.time()
    if model_name == "codellama" and model_name == "qwen2.5-coder":
        # Directly use the question for response without context
        selected_prompt_template = select_prompt_template(question)
        prompt = PromptTemplate.from_template(selected_prompt_template)
        chain_result = {"question": question}  # Only pass the question
    
    try:
        response_placeholder = st.empty()
        response = ""

        for chunk in chain.stream(chain_result):
            response += chunk  # Append each chunk to the response string
            response_placeholder.write(response)

    except Exception as e:
        response = f"An error occurred: {str(e)}"

    st.session_state.messages.append({"role": "BOT", "content": response})



    

    feedback_col = st.columns(2)
    with feedback_col[0]:
        if st.button("👍", key=f"thumbs_up_{len(st.session_state.messages) - 1}"):  # Use index for unique key
            st.session_state.messages[-1]['feedback'] = True  # Add feedback
    with feedback_col[1]:
        if st.button("👎", key=f"thumbs_down_{len(st.session_state.messages) - 1}"):  # Use index for unique key
            st.session_state.messages[-1]['feedback'] = False  # Add feedback

    query = question.lower()
    words = query.split()

    image_flag = True

    for word in words:
        if word in image_terms:
            image_flag = True

    if image_flag:
        results = img_vectorstore.similarity_search_with_score(query, k=5)

        scores_dict = {}
        result_ids = []

        for doc, score in results:
            print(score)
            if score < 500.00:
                scores_dict[doc.metadata['faiss_id']] = score
            

        sorted_dict_asc = dict(sorted(scores_dict.items(), key=lambda item: item[1]))

        for key in sorted_dict_asc.keys():
            result_ids.append(key)

        result_paths = mapping_df[mapping_df['faiss_id'].isin(result_ids)]['image_path'].to_list() 
        if result_paths:
            for i in result_paths:
            #top_path = result_paths[3]  # Get the first result
                img = Image.open(i)
                st.image(img, caption=os.path.basename(i))  # Display the top result
        else:
            pass

    model_end = time.time()
    logging.info(f"Model generated answer in {model_end - model_start:.2f} seconds")

# Download chat history button
if st.button("Download Chat History"):
    file_path = save_chat_history_with_feedback()  # Updated to include feedback
    if file_path:
        st.write(f"Chat history saved as {file_path}")
        st.download_button(
            label="Download Chat History",
            data=open(file_path, "r").read(),
            file_name=f"{st.session_state['username']}_chat_history_with_feedback.txt"  # Updated file name
        )
