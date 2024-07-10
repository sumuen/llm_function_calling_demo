from haystack import Pipeline
from haystack.nodes import FARMReader, BM25Retriever
from haystack.document_stores import InMemoryDocumentStore
from haystack.utils import fetch_archive_from_http, convert_files_to_docs, clean_wiki_text

# 创建内存文档存储
document_store = InMemoryDocumentStore()

# 下载示例文档
doc_dir = "data/tutorial1"
s3_url = "https://s3.eu-central-1.amazonaws.com/deepset.ai-farm-qa/datasets/documents.tutorial1.zip"
fetch_archive_from_http(url=s3_url, output_dir=doc_dir)

# 将文件转换为文档并清理文本
docs = convert_files_to_docs(dir_path=doc_dir, clean_func=clean_wiki_text, split_paragraphs=True)

# 写入文档存储
document_store.write_documents(docs)

# 设置BM25检索器
retriever = BM25Retriever(document_store=document_store)

# 设置FARM阅读器
reader = FARMReader(model_name_or_path="deepset/roberta-base-squad2")

# 创建Pipeline
pipe = Pipeline()
pipe.add_node(component=retriever, name="Retriever", inputs=["Query"])
pipe.add_node(component=reader, name="Reader", inputs=["Retriever"])

# 进行问答
query = "What is the capital of France?"
result = pipe.run(query=query, params={"Retriever": {"top_k": 10}, "Reader": {"top_k": 5}})

# 输出结果
for answer in result['answers']:
    print(f"Answer: {answer.answer}, Score: {answer.score}")
