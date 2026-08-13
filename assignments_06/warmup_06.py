from dotenv import load_dotenv
import os

if load_dotenv():
    print("API key loaded successfully.")
else:
    print("Warning: could not load API key. Check your .env file.")
    
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.evaluation import FaithfulnessEvaluator, RelevancyEvaluator
from llama_index.llms.openai import OpenAI
    
## =============================== RAG Concepts =====================================================

##----------------------------------- Concepts Question 1 ---------------------------------------------


"""
Scenario A — RAG:

RAG would be the best choice for this scenario because there are a lot of PDFs and they get updated
every quarter. RAG lets the AI look up the most relevant information from all the internal company
documents when someone asks a question, so we don't have to retrain the model every time the policies change.

Scenario B — Fine-tuning:

Fine-tuning would be the best choice for this scenario because they already have 3,000 examples of
their own writing produced over the years. The model can learn the company's specific writing style from
those examples and use that style when creating new product copy.

Scenario C — Prompt engineering:

Prompt engineering would be the best choice for this scenario because in this situation since the analyst only
needs to work with 1 short report. She can just give the report to the LLM in the prompt and ask her questions 
about it, so there is no real need for RAG or fine-tuning.

"""

## ---------------------------------Concepts Question 2-----------------------------------------

""" 
When AI confidently answer wrong this can be more harmful because it can make people believe the information is correct even 
when it isn't. If the AI says "I'm not sure," then the person knows they should double-check the information before using it.


Example, if someone asks an AI about a medical symptom or a medical condition and the AI confidently says that it is nothing 
serious no need to worrry at all. when it actually could be a serious condition, the person might ignore it and not get medical help.

If an AI sounds like it knows exactly what it's talking about, people may trust it without questioning the answer.
But if it admits that it isn't sure, it gives people a reason to stop and verify the information.

"""


##----------------------------------------Concepts Question 3------------------------------------------------------------


"""   


# RAG Pipeline Steps in Correct Order:

 1. Extract text from source documents
    Get the actual texts from the source documents or any other source.

 2. Split text into chunks
    Splitting the texts into smaller chunks so we can get the most relevant
    chunks rather than the whole documents.

 3. Convert text chunks into embeddings
    Converting each chunk into a numerical representation of its meaning.

 4. Receive the user's query
    Get the question or request that the user sends to the system.

 5. Embed the user's query
    Converting the user's query into a numerical representation of its meaning.

 6. Retrieve the most relevant chunks
    Getting the chunks that are most related to the user's question.

 7. Inject retrieved chunks into the prompt
    Adding the relevant chunks to the prompt so the LLM has the information.

 8. Generate a response from the LLM
    The LLM uses the question and retrieved information to create the final answer.


"""

import string


def simple_keyword_retrieval(query, documents, verbose=True):
    """Keyword retrieval using token overlap scoring."""
    stopwords = {
        "a", "an", "the", "and", "or", "in", "on", "of", "for", "to", "is",
        "are", "was", "were", "by", "with", "at", "from", "that", "this",
        "as", "be", "it", "its", "their", "they", "we", "you", "our"
    }
    translator = str.maketrans("", "", string.punctuation)

    query_words = {
        w.translate(translator)
        for w in query.lower().split()
        if w not in stopwords
    }
    if verbose:
        print(f"\nQuery tokens (filtered): {sorted(query_words)}")

    scores = []
    for name, content in documents.items():
        content_words = {
            w.translate(translator)
            for w in content.lower().split()
            if w not in stopwords
        }
        overlap = query_words & content_words
        score = len(overlap)
        scores.append((score, name, content))
        if verbose:
            print(f"[{name}] overlap={score} -> {sorted(overlap)}")

    scores.sort(reverse=True)
    best = next(((name, content) for score, name, content in scores if score > 0), None)
    if best:
        if verbose:
            print(f"\nSelected best match: {best[0]}")
        return [best]
    else:
        if verbose:
            print("\nNo overlapping keywords found.")
        return [("None found", "No relevant content.")]
    
    


##-------------------------------------------------------------Keyword Question 1------------------------------------   
    
    
query = "What are your hours on weekends?"

documents = {
    "menu.txt": "We serve espresso, lattes, cappuccinos, and cold brew. Pastries include croissants and muffins baked fresh daily. Oat milk and almond milk are available.",
    "hours.txt": "We are open Monday through Friday from 7am to 7pm. On weekends we open at 8am and close at 5pm. We are closed on Thanksgiving and Christmas Day.",
    "hiring.txt": "We are currently hiring baristas and shift supervisors. Send your resume to jobs@groundworkcoffee.com.",
    "loyalty.txt": "Join our loyalty program to earn one point per dollar spent. Redeem 100 points for a free drink of your choice.",
}

simple_keyword_retrieval(query, documents)

"""
After running the function the loyalty.txt documents got selected because hours.txt , hiring.txt ,loyalty.txt all have 
the same score of 1 and as in the simple_keyword_retrieval we have scores.sort(reverse=True) it selects the name in
alphabetically reverse order which is loyalty.txt 

"""

##----------------------------------------Keyword Question 2---------------------------------------------

query = "Do you have anything without caffeine?"

simple_keyword_retrieval(query, documents)

"""
No Documents were selected because there are no matching words from any documents or No overlapping keywords found.

Keyword RAG did not get this right because it only matches exact keywords. None of the documents contain words such as
"caffeine" or "without", so there is no keyword overlap. 

Semantic RAG would do better because it can match related meaning rather than requiring the exact same words.


"""

##---------------------------------------Keyword Question 3---------------------------------------------------------

"""
I predict that as there are no words matching in the quary and the documents the output would be No overlapping keywords found
    
"""

query = "How do I sign up for rewards?"
simple_keyword_retrieval(query, documents)

"""
Yes, It did as i predicted. As all the documents have 0 sore it doesnt select any documents.
and the prediction was correct because it just compares words and if nothing matches is score 0.

"""


###================================================Semantic RAG Concepts==================================================

##--------------------------------------------------Semantic Question 1------------------------------------------------------


"""
    
    1. Vector embedding is a numerical representation of text. LLMs cannot work with text directly as numbers, so vector embeddings 
       convert text into numbers while keeping the meaning of the text as much as possible.
       
    2. Two text chunks have cosine similarity scores of 0.85 and 0.30. The chunk with the score 0.85 is more relevant.
       the higher the score the more is the relevancy.
       
    3. Semantic search can find a relevant chunk even when none of the exact words from the query appear in the chunk because
       it compares the meaning of the query and text using their vector embeddings rather than keyword based rag.
       
    
"""


##-----------------------------------------Semantic Question 2---------------------------------------------------------------------

"""
    
    
# | Feature                 | Keyword RAG                    | Semantic RAG                              |
# | ----------------------- | ------------------------------ | ----------------------------------------- |
# | What is compared?       | Exact word overlap             | Vector embeddings                         |
# | What is retrieved?      | Full document                  | Most relevant chunks                      |
# | Can it handle synonyms? | No                             | Yes                                       |
# | Storage format          | Plain text dictionary          | Embeddings/chunks in a vector index       |
# | Relevance score         | Number of overlapping keywords | Cosine similarity score                   |

"""

##-----------------------------------------------LlamaIndex---------------------------------------------------

##-------------------------------------------LlamaIndex Question 1----------------------------------------------


print("\n  LlamaIndex Question 1")

documents = SimpleDirectoryReader("assignments_06/resources/brightleaf_pdfs").load_data()
index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine(similarity_top_k=3)


questions = [
    "What employee benefits does BrightLeaf offer?",
    "What are BrightLeaf's security policies?",
]

for q in questions:
    print("\n" + "- " * 80)
    print(f"Question: {q}")
    print("\n" + "- " * 80)
    response = query_engine.query(q)
    print(f"\nAnswer:\n{response}\n")
    print("\n" + "- " * 80)
    print("\nRetrieved source chunks:")
    
    for i, source_node in enumerate(response.source_nodes, start=1):
        print(f"\nSource {i}")
        print(f"Similarity score: {source_node.score:.4f}")
        print(f"Chunk: {source_node.node.get_content()[:150]}")
        
        
"""   

Question 1:

After observing the output of Question 1, we can say that the chunks are mostly related, but not fully. Chunk 1 is highly related 
because it talks about employee benefits, but Chunk 2 and Chunk 3 are less related to the question. Chunk 3 mostly talks about Network 
and Data Security, not employee benefits.

The model's response sounds confident enough and specific, like mentioning 401(k), and there are no phrases 
like "I'm not sure." The unexpected thing was that the last chunk it retrieved was not very relevant to the question.


Question 2:

After observing the output of Question 2, we can say that the chunks are mostly related, but not fully. 
Chunk 1 is highly relevant because it talks about Network and Data Security, but Chunk 2 and Chunk 3 are less relevant to the question.

The model's response sounds confident enough and specific, and there are no phrases like "I'm not sure." 
The unexpected thing was that the second and third chunks it retrieved were not very relevant to the question.



"""       
        
##---------------------------------------------LlamaIndex Question 2------------------------------------------

print("\n  LlamaIndex Question 2")
query_engine = index.as_query_engine(similarity_top_k=1)
q = "What employee benefits does BrightLeaf offer?"

print("\n" + "- " * 80)
print(f"Question with k=1: {q}")
print("\n" + "- " * 80)
response = query_engine.query(q)
print(f"\nAnswer:\n{response}\n")
print("\n" + "- " * 80)
print("\nRetrieved source chunks:")
    
for i, source_node in enumerate(response.source_nodes, start=1):
    print(f"\nSource {i}")
    print(f"Similarity score: {source_node.score:.4f}")
    print(f"Chunk: {source_node.node.get_content()[:150]}")
        
        
query_engine = index.as_query_engine(similarity_top_k=5)
q = "What employee benefits does BrightLeaf offer?"

print("\n" + "- " * 80)
print(f"Question with k=5: {q}")
print("\n" + "- " * 80)
response = query_engine.query(q)
print(f"\nAnswer:\n{response}\n")
print("\n" + "- " * 80)
print("\nRetrieved source chunks:")
    
for i, source_node in enumerate(response.source_nodes, start=1):
    print(f"\nSource {i}")
    print(f"Similarity score: {source_node.score:.4f}")
    print(f"Chunk: {source_node.node.get_content()[:150]}")       


        
"""  
With k = 1 and k = 5, the responses are pretty much the same. They have some different words, but mostly they are similar.
With k = 5, the model included some extra information about diversity, equity, and inclusion, while the k = 1 response was a little more focused.

This indicates that more retrieved context is not always better because the lower-ranked chunks are less relevant or not related to the 
question. In this case, k = 1 was enough to give a good answer, so adding more chunks did not make the response much better.


"""
        
##-----------------------------------------LlamaIndex Question 3----------------------------------------------------        
print("\n  LlamaIndex Question 3")
q = "What are BrightLeaf's warrenty policies for their product?"

query_engine = index.as_query_engine(similarity_top_k=3)

print("\n" + "- " * 80)
print(f"Question: {q}")
print("\n" + "- " * 80)
response = query_engine.query(q)
print(f"\nAnswer:\n{response}\n")
print("\n" + "- " * 80)
print("\nRetrieved source chunks:")
    
for i, source_node in enumerate(response.source_nodes, start=1):
    print(f"\nSource {i}")
    print(f"Similarity score: {source_node.score:.4f}")
    print(f"Chunk: {source_node.node.get_content()[:150]}")    
        
""" 

I expected the query to be difficult because the warranty information may not be in the documents. The system did not hallucinate 
and correctly said the warranty policy was not mentioned in the provided context.


The retrieved chunks were unrelated, covering employee benefits, the company overview, and network security.

I would make the system better at knowing when it does not have the answer. Maybe I could use a similarity score limit so
it does not use unrelated chunks when answering the question.


"""       
        
        
##--------------------------------------LlamaIndex Question 4----------------------------------------------

print("\n  LlamaIndex Question 4")
q = "What employee benefits does BrightLeaf offer?"

llm = OpenAI(model="gpt-4o-mini", temperature=0.2)

faithfulness_evaluator = FaithfulnessEvaluator(llm=llm)
relevancy_evaluator = RelevancyEvaluator(llm=llm)

response = query_engine.query(q)

print("\n" + "- " * 80)
print("\nQ:", q)
print("\n" + "- " * 80)
print("\nA:", response)
# Evaluate faithfulness and relevancy
faithfulness_result = faithfulness_evaluator.evaluate_response(query=q, response=response)
print("\n" + "- " * 80)
print("Faithfulness Evaluation: " + str(faithfulness_result.score))
print("\n" + "- " * 80)
relevancy_result = relevancy_evaluator.evaluate_response(query=q, response=response)
print("Relevancy Result: " + str(relevancy_result.score))
print("\n" + "- " * 80)


q = "What is BrightLeaf's policy on owning a cricket team?"

response = query_engine.query(q)

print("\n" + "- " * 80)
print("\nQ:", q)
print("\n" + "- " * 80)
print("\nA:", response)
# Evaluate faithfulness and relevancy
faithfulness_result = faithfulness_evaluator.evaluate_response(query=q, response=response)
print("\n" + "- " * 80)
print("Faithfulness Evaluation: " + str(faithfulness_result.score))
print("\n" + "- " * 80)
relevancy_result = relevancy_evaluator.evaluate_response(query=q, response=response)
print("Relevancy Result: " + str(relevancy_result.score))
print("\n" + "- " * 80)



"""  

1.  A faithfulness score of 1.0 means that the model's response is supported by the retrieved context. A score of 0.0 indicates that the
    response is not supported by the retrieved context.
    
2.  The relevancy score evaluates how closely or relevant the response aligns with the user's question. Faithfulness evaluates whether the 
    information in the response can be supported by the retrieved content.
    
3.  Yes, the score changes between the two queries. This happens because query 2 is about something for which there is no information. 
    Therefore, the model's response does not mention anything about owning a cricket team.,so both scores were 0.0.
    
    
4.  LLM-as-a-judge is when we use a diffrent LLM to score another model's answer. It is useful instead of exact-match accuracy because the same 
    question can have multiple valid answers.



"""