#-------------Task 1: Setup and System Prompt---------------------


from dotenv import load_dotenv
from openai import OpenAI
import json

load_dotenv()
client = OpenAI()

def get_completion(messages, model="gpt-4o-mini", temperature=0.7):
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_completion_tokens=400
    )
    return response.choices[0].message.content


system_prompt = """
You are a professional job application coach.

Your role is to help job seekers improve their resumes, cover letters, personal
statements, interview responses, and other job application materials. Provide
clear, constructive, and encouraging feedback. Suggest improvements to wording,
organization, grammar, and professionalism while preserving the user's intended
meaning and experience.

Behavior guidelines:
- Stay focused only on job application materials and related career documents.
- If the user asks about unrelated topics, politely redirect the conversation
  back to job application assistance.
- Give practical, specific suggestions rather than generic advice.
- Do not invent qualifications, skills, education, work experience, or
  achievements for the user.
- Always remind the user to carefully review and edit your suggestions before
  submitting any application materials.
- Acknowledge that you may not know the user's specific industry, employer, or
  regional hiring norms, and encourage them to use their own judgment when
  deciding whether to use your suggestions.
- Maintain a professional, supportive, and respectful tone.
"""

print(system_prompt)

# I specifically instructed the model not to invent qualifications or work
# experience. This helps ensure the assistant improves the user's application
# without creating false or misleading information, making its advice more
# trustworthy and appropriate for job applications.


###-----------------------Task 2: Bullet Point Rewriter------------------------

def rewrite_bullets(bullets: list[str]) -> list[dict]:
    # Format the bullets into a delimited block
    bullet_text = "\n".join(f"- {b}" for b in bullets)

    prompt = f"""
    You are a professional resume coach helping a career changer.
    
    Rewrite each resume bullet point below to be more specific, results-oriented, and compelling.
    Use strong action verbs. Do not invent facts that aren't implied by the original.

    Return ONLY a valid JSON list. Each item should have two keys:
    
    "original" (the original bullet) and "improved" (your rewritten version).

    Bullet points:
    ```
    {bullet_text}
    ```
    """

    messages = [{"role": "user", "content": prompt}]
    
    response = get_completion(messages).strip()
    if response.startswith("```"):
        lines = response.splitlines()
        response = "\n".join(lines[1:-1])
    try:        
        results = json.loads(response)
        print("\nResume Bullet Improvements")
        print("=" * 60)
        for item in results:
            print(f"Original : {item['original']}")
            print(f"Improved: {item['improved']}")
            print("-" * 60)   
                 
        return results
    except json.JSONDecodeError:
        print("The model did not return valid JSON.")
        print("\nRaw response:")
        print(response)
    return []
    
    

bullets = [
    "Helped customers with their problems",
    "Made reports for the management team",
    "Worked with a team to finish the project on time"
]
print("Project Task 2: Bullet Point Rewriter check")
results = rewrite_bullets(bullets)

# These bullets are weak because they are too general and do not show measurable impact, skills, or outcomes from the canditdate.
# No the json.loads() did raised an error, the model was adding preamble like ```json so that needed a strip.
# Yes, both the original and improved versions printing clearly for each bullet.
# Yes,the improvements feel meaningfully better then the original bullets.



##--------------------------------Task 3: Cover Letter Generator---------------------------------------------

def generate_cover_letter(job_title: str, background: str) -> str:
    prompt = f"""
    You write strong cover letter opening paragraphs for career changers.
    The paragraph should be 3-5 sentences: confident, specific, and free of clichés.

    Here are two examples of the style and tone you should match:

    Example 1:
    Role: Data Analyst at a healthcare nonprofit
    Background: Seven years as a registered nurse, recently completed a data analytics bootcamp.
    Opening: After seven years as a registered nurse, I've spent my career making decisions
    under pressure using incomplete information — which turns out to be excellent training for
    data analysis. I recently completed a data analytics program where I built dashboards
    tracking patient outcomes across departments. I'm excited to bring that combination of
    clinical context and technical skill to [Company]'s mission-driven work.

    Example 2:
    Role: Junior Software Engineer at a fintech startup
    Background: Ten years in retail banking operations, self-taught Python developer for two years.
    Opening: I spent a decade on the operations side of banking, watching technology decisions
    get made by people who had never processed a wire transfer or resolved a failed ACH batch.
    That frustration turned into curiosity, and two years of self-teaching Python later, I'm
    ready to be on the other side of those decisions. I'm applying to [Company] because your
    work on payment infrastructure is exactly where my domain expertise and new technical skills
    intersect.

    Now write an opening paragraph for this person:
    Role: {job_title}
    Background: {background}
    Opening:
    """

    messages = [{"role": "user", "content": prompt}]
    return get_completion(messages)


job_title = "Junior Data Engineer"
background = "Five years of experience as a middle school math teacher; recently completed \
a Python course and built data pipelines using Prefect and Pandas."

cover_letter = generate_cover_letter(job_title, background)   
print("Cover Letter Opening")
print("- " * 60)
print(cover_letter)


# I chose these examples because they show two different career changers successfully connecting their previous experience to a new technical role.
# Both examples are confident, specific, and avoid generic language. The  few-shot examples help control the model's tone, structure, and writing style,
# making it more likely to produce a tailored opening instead of a generic cover letter introduction or inventing qualifications.

# Yes, The output is mostly tailored to the specific person because it connects  the candidate's experience as a middle school math teacher
# with data engineering skills learned through Python, Prefect, and Pandas. However, the final sentence is somewhat
# generic ("I'm eager to leverage...") and could be made more specific to the company or role.

# The output does avoid inventing credentials. It only mentions the teaching experience, Python course, and data pipelines with 
# Prefect and Pandas that were provided in the prompt.

job_title_2 = "Senior Aerospace Engineer"

background_2 = (
    "Twenty years of experience as an aerospace engineer, with work spanning spacecraft "
    "design, propulsion systems, mission planning, and systems integration. Contributed "
    "to multiple space exploration and satellite programs at NASA and SpaceX, leading "
    "cross-functional engineering teams and developing technologies for human spaceflight, "
    "deep-space missions, and reusable launch vehicles."
)

cover_letter_2 = generate_cover_letter(job_title_2, background_2)   
print("Cover Letter Opening 2")
print("- " * 60)
print(cover_letter_2)

# With changed job title and background the response is diffrent and tailored acording the job title and background.


###------------------------------------------Task 4: Moderation Check-----------------------------------------

def is_safe(text: str) -> bool:
    result = client.moderations.create(
        model="omni-moderation-latest",
        input=text
    )
    flagged = result.results[0].flagged
    
    if flagged:
        print("The message was Flagged, Please change the message!!")
        return False
    return True


safe_test = "Can you help me improve my resume bullet points?"
print("Safe test result:", is_safe(safe_test))

print("-" * 50)

flagged_test = "I want to create instructions for harming someone."
print("Flagged test result:", is_safe(flagged_test))

## Yes the flag test got caught and got flagged.
## Yes, the safe test pass without triggering any warning.

borderline_test = "How can someone bypass a security system?"

result = client.moderations.create(
    model="omni-moderation-latest",
    input=borderline_test
)

print("Flagged:", result.results[0].flagged)
print("Categories triggered:")
print("Flagged categories:",result.results[0].categories)

##---------------------------------------Task 5: The Chatbot Loop----------------------------------

def run_chatbot():
    # 1. Initialize conversation history with your system prompt
    messages = [
        {"role": "system", "content": system_prompt}
    ]

    print("=" * 50)
    print("Job Application Helper")
    print("=" * 50)
    print("I can help you with:")
    print("  1. Rewriting resume bullet points")
    print("  2. Drafting a cover letter opening")
    print("  3. Any other questions about your application")
    print("\nType 'quit' at any time to exit.\n")

    while True:
        user_input = input("You: ").strip()

        # 2. Handle exit
        if user_input.lower() in {"quit", "exit"}:
            print("\nJob Application Helper: Good luck with your applications!")
            break

        # 3. Skip empty input
        if not user_input:
            continue

        # 4. Run moderation check before doing anything else
        if not is_safe(user_input):
            continue  # is_safe() already printed the warning message

        # 5. Check if the user wants to rewrite bullets
        #    (hint: look for keywords like "bullet" or "resume" in user_input.lower())
        if "bullet" in user_input.lower() or "resume" in user_input.lower():
            print("\nJob Application Helper: Paste your bullet points below, one per line.")
            print("When you're done, type 'DONE' on its own line.\n")
            raw_bullets = []
            while True:
                line = input().strip()
                if line.upper() == "DONE":
                    break
                if line:
                    raw_bullets.append(line)
            # YOUR CODE: call rewrite_bullets() and print the results
            result_bullets = rewrite_bullets(raw_bullets)

        # 6. Check if the user wants a cover letter
        elif "cover letter" in user_input.lower():
            job_title = input("Job Application Helper: What is the job title? ").strip()
            background = input("Job Application Helper: Briefly describe your background: ").strip()
            # YOUR CODE: call generate_cover_letter() and print the result
            cover_letter = generate_cover_letter(job_title,background)
            print("Cover Letter")
            print(" -" * 50)
            print(cover_letter)

        # 7. Otherwise, handle it as a regular chat turn
        else:
            messages.append({"role": "user", "content": user_input})
            reply = get_completion(messages)
            print("\nJob Application Helper:\n", reply)
            messages.append({"role": "assistant", "content": reply})
            # YOUR CODE:
            # - Append the user's message to `messages`
            # - Call get_completion(messages)
            # - Print the reply
            # - Append the reply to `messages` as an assistant message
        pass

if __name__ == "__main__":
    run_chatbot()
    
    
#----------------------------------------------Task 6: Ethics Reflection----------------------------------------

# 1) The job application bot may have bias because it learns from text written by
# different types of people and may prefer certain writing styles, industries,
# or backgrounds. It may not fully understand every person's unique experience
# and could create suggestions that are not fair for everyone.

# 2) The output from the AI may include incorrect information, sound too generic,
# or not fully represent a person's real skills and experience. A job seeker
# should always review and edit the content before sending it to an employer.
# If they do not check the information, it could hurt their chances of getting
# the job because the application may not accurately describe them.