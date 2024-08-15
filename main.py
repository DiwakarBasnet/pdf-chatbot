import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from dotenv import load_dotenv
load_dotenv()

#################### Generator LLM Model ####################
ACCESS_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")  # Reads .env file

model_id = "google/gemma-2b-it"
tokenizer =AutoTokenizer.from_pretrained(model_id, token=ACCESS_TOKEN)
# 4-bit quantization reduces memory and requires Nvidia GPU.
# quantization_config = BitsAndBytesConfig(load_in_4bit=True,
#                                          bnb_4bit_compute_dtype=torch.bfloat16)

model = AutoModelForCausalLM.from_pretrained(model_id,
                                             device_map="auto",
                                            #  quantization_config=quantization_config,
                                             token=ACCESS_TOKEN)
model.eval()
device = 'cuda' if torch.cuda.is_available() else 'cpu'


# LLM model inference function
def generate(question: str, context: str):
    if context == None or context == "":
        prompt = f"""Give a detailed answer to the following question. Question: {question}"""
    else:
        prompt = f"""Using the information contained in the context, give a detailed answer to the question.
            Context: {context}.
            Question: {question}"""
    chat = [{"role": "user", "content": prompt}]
    formatted_prompt = tokenizer.apply_chat_template(
        chat,
        tokenize=False,
        add_generation_prompt=True,
    )

    # Checking the type and content of formatted_prompt
    # print(f"Formatted_prompt ----------> {type(formatted_prompt), formatted_prompt}")

    inputs = tokenizer.encode(
        formatted_prompt, add_special_tokens=False, return_tensors="pt"
    ).to(device)
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs,
            max_new_tokens=250,
            do_sample=False,
        )
    response = tokenizer.decode(outputs[0], skip_special_tokens=False)
    response = response[len(formatted_prompt) :]    # remove input prompt from response
    response = response.replace("<eos>", "")        # remove eos token
    return response

# print(generate(question="How are you?", context=""))


#################### Encoder Model + Similarity Search ####################
from langchain_community.embeddings import(
    HuggingFaceEmbeddings
)

encoder = HuggingFaceEmbeddings(
    model_name = "sentence-transformers/all-MiniLM-L12-v2",
    mdoel_kwargs = {'device': "cpu"}
)


#################### Document Loader and Text Splitter ####################
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# load PDFs
loaders = [
    PyPDFLoader("/path/to/pdf/file1.pdf"),
    PyPDFLoader("/path/to/pdf/file2.pdf"),
]
pages = []
for loader in loaders:
    pages.extend(loader.load())

# Split text to chunks
text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
    tokenizer=AutoTokenizer.from_pretrained(
        "sentence-transformers/all-MiniLM-L12-v2"
    ),
    chunk_size=256,
    chunk_overlap=32,
    strip_whitespace=True,
)

docs = text_splitter.split_documents(pages)


#################### Vector Database ####################
from langchain.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy

faiss_db = FAISS.from_documents(
    docs, encoder, distance_strategy=DistanceStrategy.COSINE
)

retrieved_docs = faiss_db.similarity_search("My question", k=5)


#################### User Interface ####################
import os
import streamlit as st
from model import ChatModel
import rag_util


FILES_DIR = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "files")
)   # folder to store uploaded files


st.title("LLM Chatbot RAG Assistance")


@st.cache_resource
def load_model():
    model = ChatModel(model_id="google/gemma-2b-it", device="cuda")
    return model


@st.cache_resource
def load_encoder():
    encoder = rag_util.Encoder(
        model_name="sentence-transformers/all-MiniLM-L12-v2", device="cpu"
    )
    return encoder


model = load_model()        # load our LLM generator model once and then cache it
encoder = load_encoder()    # load our encoder model once and then cache it


def save_file(uploaded_file):
    """Helper function to save documents to disk"""
    file_path = os.path.join(FILES_DIR, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.get_buffer())
    return file_path
