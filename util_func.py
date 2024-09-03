from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from get_embedding_function import get_embedding_function

CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
You are a helpful document assistant.Your task is to provide complete and thorough answer to the question based on the
given context.
Context: {context}
Question: {question}
"""

def database_manipulation(db, reset_flag=False):
    if reset_flag:
        print("✨ Clearing Database")
        db.clear_database()
    
    # Create (or update) the data store
    documents = db.load_documents()
    chunks = db.split_documents(documents)
    db.add_to_chroma(chunks)


def query_rag(query_model, query_text: str):
    # Prepare the DB
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH,
                embedding_function=embedding_function
    )

    # Search the DB
    results = db.similarity_search_with_score(query_text, k=5)

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    
    response_text = query_model(prompt)

    return response_text
