# image_database.py
import os
from PIL import Image
import pytesseract
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain_huggingface import HuggingFaceEmbeddings

# Set the path to Tesseract executable
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def extract_text_from_image(image_path):
    """Extract text from a single image using pytesseract."""
    image = Image.open(image_path)
    text = pytesseract.image_to_string(image)
    return text

def load_text_from_images(image_dir):
    """Load text from all images in the specified directory."""
    texts = []
    for filename in os.listdir(image_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(image_dir, filename)
            text = extract_text_from_image(image_path)
            texts.append(text)
    return texts

def create_vector_db_from_texts(texts, db_path):
    """Create a FAISS vector database from the extracted texts."""
    # Save texts to a temporary file to use with TextLoader
    temp_file = 'temp_texts.txt'
    with open(temp_file, 'w') as f:
        for text in texts:
            f.write(text + '\n')

    # Use TextLoader to load documents
    loader = TextLoader(temp_file)
    documents = loader.load()

    # Initialize embeddings
    embeddings = HuggingFaceEmbeddings()

    # Create the FAISS vector database
    db = FAISS.from_documents(documents, embeddings)

    # Save the vector database to the specified path
    db.save_local(db_path)

    # Clean up temporary file
    os.remove(temp_file)

def main():
    image_dir = "./images"  # Directory containing image files
    db_path = "./image_db"  # Directory to save the vector database

    # Extract text from images
    texts = load_text_from_images(image_dir)

    # Create the vector database
    create_vector_db_from_texts(texts, db_path)
    print(f"Vector database created and saved to {db_path}")

if __name__ == "__main__":
    main()
