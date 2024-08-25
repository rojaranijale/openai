from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredExcelLoader
from langchain.vectorstores import utils as chromautils
import os
from dotenv import load_dotenv,find_dotenv
load_dotenv(find_dotenv())

class EmbeddingGenerator:
    def __init__(self, excel_file="All_ScrapedText_03_06.xlsx", chunk_size=500, chunk_overlap=50):
        self.excel_file = excel_file
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def load_papers(self):
        """Load papers from the directory and concatenate their content."""
        loader = UnstructuredExcelLoader(self.excel_file)
        docs = loader.load()
        docs = chromautils.filter_complex_metadata(docs)
        return docs

    def split_text(self, docs):
        """Split the text into manageable chunks."""
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        return text_splitter.split_documents(docs)

    def generate_embeddings(self, paper_chunks):
        """Create Qdrant vector store from paper chunks."""
        vectorstore = Chroma.from_documents(documents=paper_chunks, embedding=OpenAIEmbeddings(openai_api_key = os.environ.get("OPEN_AI_KEY")))
        return vectorstore.as_retriever()

if __name__ == "__main__":
    generator = EmbeddingGenerator()
    full_text = generator.load_papers()
    paper_chunks = generator.split_text(full_text)
    retriever = generator.generate_embeddings(paper_chunks)
    print("Embeddings generated and retriever is ready.")
