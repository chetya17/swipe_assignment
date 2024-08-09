import pandas as pd
import re
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from utils.chain import ChatWithOllama

def generate_extraction_prompt():
    return """
    Extract the following details from the payment invoice:
    1. Customer Name
    2. Billing Address
    3. Product Name
    4. Amount Payable

    Provide the details in the following format:
    - Customer Name: [Name]
    - Billing Address: [Address]
    - Product Name: [Product]
    - Amount Payable: [Amount]

    Context: {context}
    """

def extract_invoice_details_from_response(response):
    details = {
        "Customer Name": re.search(r'Customer Name: (.+)', response).group(1).strip(),
        "Billing Address": re.search(r'Billing Address: (.+)', response).group(1).strip(),
        "Product Name": re.search(r'Product Name: (.+)', response).group(1).strip(),
        "Amount Payable": re.search(r'Amount Payable: (.+)', response).group(1).strip(),
    }
    return details

def main():
    # Initialize FAISS vector database
    embeddings = HuggingFaceEmbeddings()
    vector_db_path = "../database_image/image_db"  # Path to your existing vector database
    db = FAISS.load_local(vector_db_path, embeddings,allow_dangerous_deserialization=True)

    # Initialize RAG application
    chat_bot = ChatWithOllama(multi_retrival=False)

    # Query the vector database for context
    prompt_for_retrieval = "Provide the context for extracting the following details: Customer Name, Customer Address, Product Name, Amount Payable."
    context_chunks = db.similarity_search(prompt_for_retrieval)

    # Combine the text chunks into a single context
    context_text = "\n".join([chunk.page_content for chunk in context_chunks])

    # Generate and execute prompt
    prompt = generate_extraction_prompt().format(context=context_text)
    response = ''.join(chat_bot.GetResponse(prompt))  # Combine all response chunks

    # Extract details
    details = extract_invoice_details_from_response(response)

    # Create DataFrame and save to CSV
    df = pd.DataFrame([details])
    csv_file_path = "invoice_details_from_image.csv"
    df.to_csv(csv_file_path, index=False)
    print(f"Data saved to {csv_file_path}")

if __name__ == "__main__":
    main()
