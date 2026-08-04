from dotenv import load_dotenv
from openai import OpenAI
import json

#---------------------------Part 1: Warmup Exercises------------------------------------

#-----------------------------API Question 1--------------------------------------------

load_dotenv()
client = OpenAI()

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "What is one thing that makes Python a good language for beginners?"}]
)

print("Response:")
print(response.choices[0].message.content)

print("\nModel:")
print(response.model)

print("\nTotal Tokens:")
print(response.usage.total_tokens)



#-----------------------------API Question 2-------------------------------------

temperatures = [0, 0.7, 1.5]
for t in temperatures:
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Suggest a creative name for a data engineering consultancy."}],
        n=1,
        temperature=t)
    print(f"\nResponse with Temperature {t}:")
    print(response.choices[0].message.content)
    
# # I noticed that Response with Temperature 1.5 have 10 suggested names but other two have 1 response.
# # it shows shows the flexiblity in high temperature
# # If i needed a  consistent, reproducible output i will thake temperature 0 


# #-----------------------------API Question 3--------------------------


response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Give me a one-sentence fun fact about pandas (the animal, not the library)."}],
    n=3,
    temperature=1.0
)

for choice in response.choices:
    print(" -"* 50) 
    print(choice.message.content)


# #----------------------------------API Question 4--------------------------------------------


response = client.chat.completions.create(
    model="gpt-4o-mini",
    max_tokens=15, 
    messages=[{"role": "user", "content": "Explain how neural networks work."}],
    n=3,
    temperature=1.0
)
print(" -"* 50)
print(response.choices[0].message.content)

# After Using the max_token=15 the response got cut off just some few begginig words which stopped the response
# when token limit reached
# In real life applications I want to use max tokens to keep costs in limit, or limit the response size.



#--------------------------------System Messages and Personas------------------------------------------

#-------------------------------System Question 1--------------------------------------


response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{
        "role": "system", "content": "You are a patient, encouraging Python tutor. You always explain things simply and end with a word of encouragement.",
        "role": "user", "content": "I don't understand what a list comprehension is."
        }],
    n=1
)
print(" -"* 50)
print(response.choices[0].message.content)

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{
        "role": "system", "content": "You are a bad-tempered, rude, sarcastic, arrogant, impatient, and short-tempered boss. You use casual slang, give blunt answers, and act annoyed, but you still provide accurate information.",
        "role": "user", "content": "I don't understand what a list comprehension is."
        }],
    n=1
)
print(" -"* 50)
print(response.choices[0].message.content)

# ----------------------------------------------System Question 2----------------------------------------------
 
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "My name is Jordan and I'm learning Python."},
    {"role": "assistant", "content": "Nice to meet you, Jordan! Python is a great choice. What would you like to work on?"},
    {"role": "user", "content": "Can you remind me what my name is?"}
]

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=messages,
    n=1
)
print(" -"* 50)
print(response.choices[0].message.content)

# The model know Jordan's name, even though it's stateless because we are passing all the previous user
# assistant messages. 


#------------------------------------------------Prompt Engineering-----------------------------------------------

#-----------------------------------------------Prompt Question 1 — Zero-Shot----------------------------------------

reviews = [
    "The onboarding process was smooth and the team was welcoming.",
    "The software crashes constantly and support never responds.",
    "Great price, but the documentation is nearly impossible to follow."
]

for i, review in enumerate(reviews):
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Classify the sentiment of the following review as positive, negative, or mixed. Respond with only the sentiment."},
            {"role": "user", "content": review}
        ],
        n=1
    )
    print(f"Review {i}: {response.choices[0].message.content}")
    
    
# #-----------------------------------Prompt Question 2 — One-Shot---------------------------------

print("\nPrompt Question 2 — One-Shot")
for i, review in enumerate(reviews):
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Classify the sentiment of the following review as positive, negative, or mixed. Respond with only the sentiment."},
            {"role": "user",
             "content": f"""
              Example:
              Review: "Fast shipping but the item arrived damaged."
              Sentiment: mixed
              Now classify this review:
              Review: "{review}"
              Sentiment:
             """}
        ],
        n=1
    )
    print(f"Review {i}: {response.choices[0].message.content}")
    
# # Adding one example did not change the format or consistency of the output compared to Q1
# # Both time the out put were the same


# #--------------------------------------------Prompt Question 3 — Few-Shot----------------------------------------------

print("\nPrompt Question 3 — Few-Shot")
for i, review in enumerate(reviews):
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Classify the sentiment of the following review as positive, negative, or mixed. Respond with only the sentiment."},
            {"role": "user",
             "content": f"""
              Examples:
              
              Review: "Fast shipping but the item arrived damaged."
              Sentiment: mixed
              
              Review: "The product works perfectly and the customer service was excellent."
              Sentiment: positive
              
              Review: "The app is slow, crashes often, and wastes my time."
              Sentiment: negative
              
              Now classify this review:
              Review: "{review}"
              Sentiment:
             """}
        ],
        n=1
    )
    print(f"Review {i}: {response.choices[0].message.content}")
    
    
# Zero-shot prompting gives only the task instructions and is useful when the task
# is simple or the model already understands the expected output format.

# One-shot prompting adds one example to demonstrate the desired format. It is
# useful when we want more consistent formatting or need to guide the model
# without adding much prompt length.

# Few-shot prompting provides multiple examples covering different cases. It is
# useful when the task is more complex, when accuracy matters, or when the model
# needs clearer guidance about categories and edge cases.

###---------------------------------Prompt Question 4 — Chain of Thought------------------------------------


u_content ="""A data engineer earns $85,000 per year. She gets a 12% raise, then 6 months
later takes a new job that pays $7,500 more per year than her post-raise salary. What is her final annual salary?"""

response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "show its reasoning step by step before giving a final answer"},
            {"role": "user", "content": u_content}
        ],
        n=1
    )
print(f"{response.choices[0].message.content}")



####--------------------------------------------------Prompt Question 5 — Structured Output-----------------------------------------


review = "I've been using this tool for three months. It handles large datasets well, \
but the UI is clunky and the export options are limited."

response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Analyze the review and return only valid JSON. "
                      "The JSON must contain exactly these keys: "
                      "sentiment, confidence, and reason. "
                      "confidence must be a float between 0 and 1. "
                      "reason must be one sentence."},
            {"role": "user", "content": review}
        ],
        n=1
    )

raw_response = response.choices[0].message.content
print("Raw response:")
print(raw_response)
try:
    result = json.loads(raw_response)
    print("\nParsed fields:")
    print("Sentiment:", result["sentiment"])
    print("Confidence:", result["confidence"])
    print("Reason:", result["reason"])

except json.JSONDecodeError:
    print("\nError: Response was not valid JSON.")
    print("Raw response for debugging:")
    print(raw_response)
    
    
###----------------------------------Prompt Question 6 — Delimiters---------------------------------------------

user_text = "First boil a pot of water. Once boiling, add a handful of salt and the \
pasta. Cook for 8-10 minutes until al dente. Drain and toss with your sauce of choice."

prompt = f"""
You will be given text inside triple backticks.
If it contains step-by-step instructions, rewrite them as a numbered list.
If it does not contain instructions, respond with exactly: "No steps provided."

```{user_text}```
"""


response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
                {"role": "user", "content": prompt}
                ],
        n=1
    )

print("First Prompt Response:")
print(response.choices[0].message.content)

user_text_2 = "Sky is blue so does the ocean \
tree is green and flowers are many colors."

prompt_2 = f"""
You will be given text inside triple backticks.
If it contains step-by-step instructions, rewrite them as a numbered list.
If it does not contain instructions, respond with exactly: "No steps provided."

```{user_text_2}```
"""

response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
                {"role": "user", "content": prompt_2}
                ],
        n=1
    )

print("Second Prompt Response:")
print(response.choices[0].message.content)





###-------------------------------------------Local Models with Ollama-----------------------------------------

##-----------------------------------Ollama Question 1---------------------------------------------

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Explain what a large language model is in two sentences."}]
)

print("Response:")
print(response.choices[0].message.content)


# ollama response

# A large language model is an artificial intelligence system trained on massive datasets to understand and generate
# human-like text, enabling tasks like writing, answering questions, or creating content. It processes vast amounts
# of information, understands context, and can generate coherent responses, making it highly versatile in various applications.

# I notice some wording diffrences but overall both explains about large language model in 2 sentences
# One advantage of running model locally is we dont need to worry about tokens and costs.
# One disadvantage of running model locally is its not as efficient and accurate as cloud models