from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI

# 设置你的 OpenAI API 密钥
import os
os.environ["OPENAI_API_KEY"] = "YOUR_OPENAI_API_KEY"  # 替换为你的 OpenAI API Key

# 1. 加载文档
loader = TextLoader("your_document.txt")  # 替换为你的文档路径
documents = loader.load()

# 2. 分割文本
text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
texts = text_splitter.split_documents(documents)

# 3. 创建嵌入和向量存储
embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(texts, embeddings)

# 4. 创建检索器
retriever = vectorstore.as_retriever()

# 5. 创建 QA 链
qa = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type="stuff", retriever=retriever)

# 6. 提问并获取答案
query = "What is the main topic of this document?"  # 替换为你的问题
answer = qa.run(query)

print(f"Question: {query}")
print(f"Answer: {answer}")