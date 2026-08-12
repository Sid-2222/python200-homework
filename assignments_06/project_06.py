##-------------------------------------------------------Step 1: Setup-------------------------------------------------------------


from pathlib import Path
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex


if load_dotenv():
    print("✅ Successfully loaded API key")
else:
    print("⚠️ Failed to load API key from .env file")


docs_dir = Path("assignments_06/resources/groundwork_docs")
assert docs_dir.exists(), f"Document directory not found: {docs_dir}"


##------------------------------------Step 2: Load the Documents----------------------------------------------------------------
    
docs = SimpleDirectoryReader("assignments_06/resources/groundwork_docs").load_data()

print(f"\nNumber of documents loaded: {len(docs)}")
print("\n" + "- " * 80)
for d in docs:
    print(d.metadata["file_name"])
    
    
index = VectorStoreIndex.from_documents(docs)
query_engine = index.as_query_engine(similarity_top_k=3)

print("\n" + "- " * 80)
print("Index built successfully. Ready to answer questions.")



##--------------------------------------------Step 4: Query the Assistant---------------------------------------------------------


questions = [
    "What are Groundwork's hours on weekends?",
    "Do you offer any dairy-free milk options?",
    "How does the loyalty program work?",
    "How did Groundwork Coffee get started?",
    "Do you offer catering or wholesale orders?",
]


for question in questions:
    response = query_engine.query(question)

    print("\n" + "- " * 80)
    print(f"Question: {question}")
    print(f"Answer: {response}")
    
    
    if response.source_nodes:
            top_node = response.source_nodes[0]

            document_name = top_node.metadata.get("file_name", "Unknown")
            similarity_score = top_node.score
            chunk_text = top_node.text[:200]
            print("\n" + "- " * 80)
            print(f"\nSource document: {document_name}")
            print(f"\nSimilarity score: {similarity_score}")
            print(f"\nChunk text: {chunk_text}")
    else:
            print("No source nodes were retrieved.")
            
            
            
""" 

 Overall, the assistant sounded confident and accurate. Most of the answers  were directly supported by the retrieved source documents.
 The answers about weekend hours, how Groundwork Coffee started, and catering/wholesale orders matched the retrieved chunks very closely.
 Overall, the responses were confident and useful.


"""
            

##----------------------------------Step 5: Find a Failure---------------------------------

q = "What was Groundwork Coffee's revenue in 2020?"

response = query_engine.query(q)
print("\n" + "- " * 80)
print(f"Question: {q}")
print(f"\nFull response:\n{response}")
print("\n" + "- " * 80)

print("Retrieved source nodes:")



for i, node in enumerate(response.source_nodes[:3], start=1):
    document_name = node.metadata.get("file_name", "Unknown")
    similarity_score = node.score
    chunk_text = node.text[:200]

    print("\n" + "- " * 80)
    print(f"\nSource Node {i}")
    print(f"\nDocument name: {document_name}")
    print(f"\nSimilarity score: {similarity_score}")
    print(f"\nChunk text: {chunk_text}")
    print("\n" + "- " * 80)
    
    
"""  
 I asked about Groundwork Coffee's revenue in 2020 because I expected this information to be not available in the documents.

The model gave the correct response and did not make up any revenue number.But the retrieved chunks were not related to revenue. 
They were about the company story, wholesale/catering, and the menu.

The model sounded less confident and clearly said the information could not be determined. This shows we should not blindly trust AI answers.

I would make the search better so it can find the right information. I would also add a limit so that unrelated information is not given to the model.


"""


##--------------------------------------------------------Step 6: Reflection-----------------------------------------------


"""   

1.  The LlamaIndex implementation took much fewer lines of code compared to building semantic RAG manually. This shows that a framework 
    like LlamaIndex can save time and make it easier to build RAG systems without writing everything from scratch.

2.  One useful use case would be for a school or college. Students could ask questions about school policies, courses, fees, or admission
    information, and the RAG system could find the answers from the school's documents.

3.  One failure RAG cannot fully prevent is the model giving a wrong or made-up answer even when the correct information was retrieved.
    The model may misunderstand the retrieved information or combine it incorrectly.




"""