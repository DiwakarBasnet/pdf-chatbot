import os
import shutil
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from get_embedding_function import get_embedding_function
from langchain_chroma import Chroma


class DataBase():
    def __init__(self):
        self.data_path = "data"
        self.chroma_path = "chroma"


    def load_documents(self):
        document_loader = PyPDFDirectoryLoader(self.data_path)
        return document_loader.load()

    def split_documents(self, documents: list[Document]):
        text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=10000,
                chunk_overlap=70,
                length_function=len,
                is_separator_regex=False,
        )
        return text_splitter.split_documents(documents)

    def add_to_chroma(self, chunks: list[Document]):
        # Load the existing DataBase
        db = Chroma(
                persist_directory=self.chroma_path, embedding_function=get_embedding_function()
        )
        # Calculate page IDs
        chunks_with_ids = self.calculate_chunk_ids(chunks)
        
        # Add or update the documents
        existing_items = db.get(include=[])     # IDs are always included by default
        existing_ids = set(existing_items["ids"])
        print(f"Number of existing documents in DB: {len(existing_ids)}")

        # Only add documents that don't exist in the DataBase
        new_chunks = []
        for chunk in chunks_with_ids:
            if chunk.metadata["id"] not in existing_ids:
                new_chunks.append(chunk)

        if len(new_chunks):
            print(f"👉 Adding new documents: {len(new_chunks)}")
            new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
            db.add_documents(new_chunks, ids=new_chunk_ids)
        else:
            print("✅ No new documents to add")

    def calculate_chunk_ids(self, chunks):
        # This will create IDs like "data/monopoly.pdf:6:2"
        # Page source : Page number : Chunk index
        last_page_id = None
        current_chunk_index = 0

        for chunk in chunks:
            source = chunk.metadata.get("source")
            page = chunk.metadata.get("page")
            current_page_id = f"{source}:{page}"

            # If the page ID is the same as the last one, increment the index
            if current_page_id == last_page_id:
                current_chunk_index += 1
            else:
                current_chunk_index = 0

            # Calculate the chunk id
            chunk_id = f"{current_page_id}:{current_chunk_index}"
            last_page_id = current_page_id

            # Add it to the page metadata
            chunk.metadata["id"] = chunk_id

        return chunks

    def clear_database(self):
        if os.path.exists(self.chroma_path):
            shutil.rmtree(self.chroma_path)


