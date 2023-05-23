from flask import Flask, request, render_template, session
from io import BytesIO
from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI

import os

app = Flask(__name__)
# Generate a random secret key
secret_key = os.urandom(16)
app.secret_key = secret_key
api_key = "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"
openai = OpenAI(openai_api_key=api_key)
# Path of the PDF file
pdf_paths = [
    os.path.join(os.getcwd(), "faqdc.pdf"),
    os.path.join(os.getcwd(), "faqmd.pdf"),
    os.path.join(os.getcwd(), "DC Real Estate Practices Draft.pdf"),
    os.path.join(os.getcwd(), "VA Real Estate Practices Draft.pdf"),
    os.path.join(os.getcwd(), "Maryland Real Estate Practices Draft.pdf"),
    os.path.join(os.getcwd(), "firstam.pdf"),
]

# Read the PDF files and concatenate the raw text
raw_text = ''
for pdf_path in pdf_paths:
    with open(pdf_path, 'rb') as f:
        pdf_data = f.read()
    reader = PdfReader(BytesIO(pdf_data))

    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text:
            raw_text += text

# Split the text into smaller chunks
text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=1500,
    chunk_overlap=200,
    length_function=len,
)
texts = text_splitter.split_text(raw_text)

# Download embeddings from OpenAI
embeddings = OpenAIEmbeddings(openai_api_key='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')

# Create an FAISS vector store for the text chunks
docsearch = FAISS.from_texts(texts, embeddings)

# Load a question answering chain
chain = load_qa_chain(openai, chain_type="stuff")
# ...

@app.route('/', methods=['GET', 'POST'])
def index():
    messages = []

    if request.method == 'POST':
        question = request.form['question']
        
        # Check if the question is PDF-based or general
        if is_pdf_based_question(question):
            
            # PDF-based question
            docs = docsearch.similarity_search(question)
            answer = chain.run(input_documents=docs, question=question)
            messages.append({'role': 'user', 'content': question})
            messages.append({'role': 'bot', 'content': answer})
        else:
            print("GPTCHQT ")
            # General question
            messages.append({'role': 'user', 'content': question})

            # Send the conversation to the language model
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=messages
            )

            # Extract the bot's reply from the response
            bot_reply = response['choices'][0]['message']['content']
            messages.append({'role': 'bot', 'content': bot_reply})
        

    # Load previous messages from session or create an empty list
    prev_messages = session.get('messages', [])
    # Concatenate previous messages with the new ones
    messages = prev_messages + messages
    # Save messages to session
    session['messages'] = messages

    return render_template('index.html', messages=messages)

# ...

def is_pdf_based_question(question):
    # Implement your logic to determine if a question is PDF-based or general
    # For example, you can check for specific keywords or patterns in the question
    # and return True if it matches the criteria for a PDF-based question.
    return True  # Placeholder implementation

# ...

if __name__ == '__main__':
    app.run(debug=True)
