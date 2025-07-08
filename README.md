# Personal_Guide
🧠 AI-Powered Document Chatbot
A simple, interactive chatbot that lets you upload PDF or web-based documents and ask questions about their content using natural language.It also provide solution to your text based queries. Powered by LangChain, Ollama, and Streamlit.

🚀 Features
📄Answer your text based queries

📄 Upload PDFs or extract content from URLs

✂️ Automatic text chunking and embedding

🔍 Semantic search with vector stores (e.g., FAISS)

💬 Chat interface to query your documents

⚡ Local or cloud LLM support (e.g., Ollama, etc.)

🛠️ Tech Stack
Component	Technology
Language Model Ollama (LLMs)
UI Framework	Streamlit
Embedding	OpenAI Embeddings 
Vector Store	FAISS
Document Parsing	PyPDF 
Prompt Chaining	LangChain

📂 Project Structure
📦 Guide
│
├── bot.py                  # Streamlit app entry point
├── .env                    # API keys and environment variables
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
✅ Requirements
 
 Python 3.10+
.env file with your keys (if using OpenAI):

# Install dependencies
pip install -r requirements.txt

▶️ Run with Ollama (Local Model)
Install Ollama:
Use LLMs(llama2)

▶️ Running the App
streamlit run app.py

📌 To-Do
Add support for multi-document querying

Enhance prompt templates

Save chat history

Add citation sources

🧑‍💻 Author
Name: Harkaranjit Kaur

GitHub: github.com/Harkarankaur

