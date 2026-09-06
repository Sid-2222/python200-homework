
#----------------------------------------------ML vs. LLM in Pipelines----------------------------------------------#

#----------------------------------------------ML/LLM Question 1-----------------------------------------------#

""" 
In this week's pipeline our ML classifier takes 4 input and it's already trained on weather data. After taking the input it
predicts whether it will rain or not. It gives 0,1.

LLM using the models prediction produce one sentence output describing the day condition for running.

The ML classifier can only generate numbers not a human readable sentence. So we use the ML for the prediction and
then using the prediction we generate a human readable sentence using the LLM.

If we swap the order the pipeline would not work as well. The LLM would be able to generate numbers but its output may be less consistent,
harder to validate, more expensive, and less predictable than our ML classifier.

But, the ML classifier is not able to generate human readable sentences. It only can generate numbers.

"""


#----------------------------------------------ML/LLM Question 2-------------------------------------------------#

"""  


Converting a date string like "2023-07-04" to day-of-week
I would use deterministic code because the result can be calculated exactly using a date library locally.

Classifying a job posting as "entry-level", "mid-level", or "senior" based on freeform text
I would use an LLM because it can understand and interpret the meaning and context of freeform job-posting text.

Predicting customer churn given 15 numeric features and a labeled training dataset
I would use an ML model because it can learn patterns from the labeled training dataset and make predictions based on the numeric features.


Normalizing inconsistent city names ("NYC", "New York City", "New York, NY") to a canonical form
I would use deterministic code because known variations can be mapped reliably to one canonical city name.

Summing a column of revenue figures
I would use deterministic code because the result can be calculated using a mathematical operation.



"""


#----------------------------------------------ML/LLM Question 3-------------------------------------------------#

"""
Incremental processing means the pipeline only processes new or changed records instead of processing everything again.

This is important because it saves time and reduces cost. If the script processed all 365 records every time, it would do the same work again,
which would make the pipeline slower and more expensive.

It could also cause data problems like duplicate records or overwriting data that was already processed.

"""

#----------------------------------------------Prompt Design-----------------------------------------------#

#--------------------------------------------Prompt Question 1---------------------------------------------#


""" 

ALTERNATIVE_SYSTEM_PROMPT = (
    "You are writing a two-sentence running recommendation for a daily weather summary app. "
    "You will receive weather conditions for a single day and a machine learning prediction "
    "about whether the day is good for running. "
    "Write exactly two sentences — direct, practical, and specific to the conditions. "
    "where the first sentence states the prediction and the second sentence explains the reasoning. "
    "Do not use bullet points, headers, or phrases like 'Based on the data'."
)


In the validation logic, I need to change the lenth check to  "if len(sentences) > 3" because the prompt
is asking no more than two sentences.

"""

#--------------------------------------------Prompt Question 2---------------------------------------------#



import time

def call_with_retry(client, messages, max_retries=3):
    for attempt in range(max_retries):
        try:
            return client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                max_tokens=100,
            )
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")

            if attempt < max_retries - 1:
                time.sleep(2)

    return None

# In a production pipeline, I would use this to handle temporary API or network failures so the pipeline 
# can retry automatically instead of failing immediately, while still giving up after 3 unsuccessful attempts.