import PyPDF2
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document

def process_pdf(pdf_file, filename):
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    
    documents = []
    for page_num in range(len(pdf_reader.pages)):
        page = pdf_reader.pages[page_num]
        text = page.extract_text()
        documents.append(Document(
            page_content=text, 
            metadata={
                "page": page_num + 1,
                "source": filename,  
            }
        ))
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=120,
        length_function=len,
        is_separator_regex=False,
    )
    chunks = text_splitter.split_documents(documents)
    
    return calculate_chunk_ids(chunks, filename)

def calculate_chunk_ids(chunks, filename):
    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = filename
        page = chunk.metadata.get("page", 0)
        current_page_id = f"{source}:{page}"

        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id

        chunk.metadata["id"] = chunk_id
        chunk.metadata["source"] = source

    return chunks