from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.document_loaders import UnstructuredPDFLoader
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain

# 上传PDF文件
loader = UnstructuredPDFLoader("./张一鸣微博2886条.pdf")
pages = loader.load_and_split()
embeddings = OpenAIEmbeddings()
docsearch = Chroma.from_documents(pages, embeddings).as_retriever()

# 输入要询问的问题
query = "以下哪一条更可能是作者所创办公司的使命？请简述原因 1）激发创造 丰富生活 2） 用科技让复杂的世界更简单 3） 帮大家吃得更好，生活更好 4） 让全球多一点幸福"
docs = docsearch.get_relevant_documents(query)
chain = load_qa_chain(OpenAI(temperature=0), chain_type="stuff")
output = chain.run(input_documents=docs, question=query)
print(output)