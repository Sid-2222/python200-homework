from pathlib import Path
from openai import OpenAI
import matplotlib
matplotlib.use('Agg')
import pandas as pd
import matplotlib.pyplot as plt
from dotenv import load_dotenv
import os
from datetime import datetime
import json
from scipy.stats import pearsonr
# smolagents imports
from smolagents import ToolCallingAgent, OpenAIServerModel, tool
from smolagents import CodeAgent

if load_dotenv():
    print("Successfully loaded environment variables from .env")
else:
    print("Warning: could not load environment variables from .env")
api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI()
print('OpenAI client created.')
RESOURCES_DIR = Path("assignments_07/resources")
















##--------------------------------------Lesson 02: Tool Definitions and the ReAct Loop------------------------------------------


##--------------------------------------Q1---------------------------------------------

def celsius_to_fahrenheit(celsius: float) -> str:
    """Convert a Celsius temperature to Fahrenheit and return it as a formatted string."""
    fahrenheit = (celsius * 9 / 5) + 32
    return f"{celsius}°C is {fahrenheit}°F"

celsius_to_fahrenheit_schema = {
        "type": "function",
        "function":{
             "name": "celsius_to_fahrenheit",
             "description":"Convert a Celsius temperature to Fahrenheit and return it as a formatted string",
             "parameters" : {
                 "type": "object",
                 "properties":{
                     "celsius": {
                         "type" : "number",
                         "description": "The temperature in degrees Celsius."
                     }
                 },
                 "required": ["celsius"],
             }
        }
    }


print(celsius_to_fahrenheit(0))
print(celsius_to_fahrenheit(100))
print(celsius_to_fahrenheit(-40))


















##------------------------------------------------------Q2---------------------------------------------------------

tools = [
    {
        'type': 'function',
        'function': {
            'name': 'get_current_time',
            'description': 'Returns the current local time as a string.',
            'parameters': {
                'type': 'object',
                'properties': {},
                'required': [],
            },
        },
    },
    {
        "type": "function",
        "function":{
             "name": "celsius_to_fahrenheit",
             "description":"Convert a Celsius temperature to Fahrenheit and return it as a formatted string",
             "parameters" : {
                 "type": "object",
                 "properties":{
                     "celsius": {
                         "type" : "number",
                         "description": "The temperature in degrees Celsius."
                     }
                 },
                 "required": ["celsius"],
             }
        }
    },
]

def get_current_time() -> str:
    '''Return the current local time as a formatted string.'''
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')


def run_agent(user_prompt: str) -> str:
    '''Run a minimal ReAct-style agent for a single user prompt.'''

    SYSTEM_PROMPT = '''You are a simple assistant that can tell the current time.
                     Use the tool get_current_time whenever a user asks about the time.'''
    
    # Step 1: start the conversation with system and user messages
    messages = [
        {'role': 'system', 'content': SYSTEM_PROMPT},
        {'role': 'user', 'content': user_prompt},
    ]

    # Step 2: first API call - the model decides whether to call a tool
    first_response = client.chat.completions.create(
        model='gpt-4.1-mini',
        messages=messages,
        tools=tools,
        tool_choice='auto',  # model chooses whether to use a tool
    )

    print("First response received from model...")
    print(first_response)
    first_message = first_response.choices[0].message

    # Record what the model said so far
    messages.append(
        {
            'role': 'assistant',
            'content': first_message.content,
            'tool_calls': first_message.tool_calls,
        }
    )

    # Step 3: check if the model requested any tools
    if first_message.tool_calls:
        print("Agentic mode engaged...")
        for tool_call in first_message.tool_calls:
            function_name = tool_call.function.name
            # In this example we only have one tool: get_current_time
            if function_name == 'get_current_time':
                tool_result = get_current_time()                
            
            elif function_name == 'celsius_to_fahrenheit':
                arguments = json.loads(tool_call.function.arguments)
                tool_result = celsius_to_fahrenheit(arguments['celsius'])
            else:
                tool_result = f'Error: unknown tool {function_name}.'

            # Print for debugging so we can see what happened
            print('Tool called:', function_name)
            print('Tool result:', tool_result)

            # Step 3b: append the tool output so the model can see it
            messages.append(
                {
                    'role': 'tool',
                    'tool_call_id': tool_call.id,
                    'name': function_name,
                    'content': tool_result,
                }
            )

        # Step 4: second API call - model sees the tool result and gives final answer
        second_response = client.chat.completions.create(
            model='gpt-4.1-mini',
            messages=messages,
        )
        print("Second response received from model...")
        print(second_response)

        final_message = second_response.choices[0].message
        return final_message.content or ''
    else:
        print("No tools needed....")

    # If there were no tool calls, the first response was already the final answer
    return first_message.content or ''

print("Question 2")

"""
1. I assume calling run_agent("Convert 100 degrees Celsius to Fahrenheit") won't trigger any tool because the llm already
   knows the formula to convert 100 degree Celsius. But the question 2 says to copy the run_agent function from the lesson
   not the tools of that lesson and the function get_current_time. so i cpoied that too. 
   
   and also it uses the tools from question 1 not from the lesson's tools of the run_agent function source.
   
2. I assume 1 API call will be enough to answer the quary without calling any tools.  

"""
answer = run_agent("Convert 100 degrees Celsius to Fahrenheit")
print(answer)

""" 
My prediction was wrong. It does call for a tool which resulted in Error: unknown tool because the question didn't mention
to add the tool celsius_to_fahrenheit in the question yet.

Also it called the API 2 times.

"""








###---------------------------------------------------Q3-----------------------------------------------------------------
 
print("\n Question 3")
response_a = run_agent("What is 37 degrees Celsius in Fahrenheit?")
print("Response A:", response_a)

"""
For response A the tool which got called is celsius_to_fahrenheit because the pronpt was What is 37
degrees Celsius in Fahrenheitwhich match the tools.

"""

response_b = run_agent("What is the boiling point of water in plain English?")
print("Response B:", response_b)

""" 
For response B there was no tool needed and the llm was able to answer from its reasoning.

"""














###=-----------------------------------------------Lesson 03: Multi-Tool Agent-----------------------------------------


class CsvManager:
    def __init__(self, resources_dir: Path):
        self.resources_dir = resources_dir
        self.df = None
        self.csv_name = None

    # --- Small internal helpers --------------------------------------

    def _normalize_csv_name(self, filename: str) -> str:
        if not filename.lower().endswith(".csv"):
            return filename + ".csv"
        return filename

    def _available_csv_files(self) -> list[str]:
        if not self.resources_dir.exists():
            return []
        return sorted(
            [
                p.name
                for p in self.resources_dir.iterdir()
                if p.is_file() and p.suffix.lower() == ".csv"
            ]
        )

    def _ensure_loaded(self):
        if self.df is None:
            files = self._available_csv_files()
            example = files[0] if files else "your_file.csv"
            return {
                "error": (
                    "No CSV is loaded yet. First load one from resources/. "
                    f"For example: load_csv '{example}'."
                )
            }
        return None

    # --- Tools (public methods) --------------------------------------

    def list_csv_files(self):
        """
        List available CSV files in resources/.
        """
        files = self._available_csv_files()
        if not files:
            return {
                "message": (
                    "No CSV files found in resources/. "
                    "Create a resources/ folder and put one or more .csv files inside it."
                ),
                "files": [],
            }
        return {"files": files}

    def load_csv(self, filename: str):
        """
        Load a CSV file from resources/ and make it the active dataset.

        filename can be "bike_commute" or "bike_commute.csv".
        """
        filename = self._normalize_csv_name(filename)
        path = self.resources_dir / filename

        if not path.exists():
            return {
                "error": f"Could not find '{filename}' in resources/.",
                "available_files": self._available_csv_files(),
            }

        self.df = pd.read_csv(path)
        self.csv_name = filename

        return {
            "message": f"Loaded {filename} with shape {self.df.shape}.",
            "columns": self.df.columns.tolist(),
        }

    def get_columns(self):
        """
        Return column names for the currently loaded CSV.
        """
        error = self._ensure_loaded()
        if error:
            return error
        return self.df.columns.tolist()

    def summarize_columns(self, columns: list[str] | None = None):
        """
        Return basic summary stats for one or more columns.

        If columns is None, summarize all columns.
        Uses pandas.describe(include="all") to stay simple and readable.
        """
        error = self._ensure_loaded()
        if error:
            return error

        if columns is None:
            data = self.df
        else:
            missing = [c for c in columns if c not in self.df.columns]
            if missing:
                return {"error": f"These columns are not in the data: {missing}"}
            data = self.df[columns]

        summary = data.describe(include="all").transpose().round(3)
        return summary.to_dict()

    def describe_column(self, column: str):
        """
        Simple summary for a single column using pandas.describe().
        """
        error = self._ensure_loaded()
        if error:
            return error

        if column not in self.df.columns:
            return {"error": f"'{column}' is not a column. Options: {self.df.columns.tolist()}"}

        s = self.df[column]
        summary = s.describe().to_dict()

        cleaned = {}
        for key, value in summary.items():
            if isinstance(value, (int, float)):
                cleaned[key] = round(value, 3)
            else:
                cleaned[key] = value

        return cleaned

    def plot_data(self, y: str, x: str | None = None, plot_type: str = "line"):
        """
        Plot from the active CSV.
    
        - If x is None: plot y vs row index.
        - If x is provided: plot y vs x.
        """
        error = self._ensure_loaded()
        if error:
            return error
    
        if plot_type not in ["scatter", "line"]:
            return "Error: I can only do 'scatter' or 'line'."
    
        if y not in self.df.columns:
            return f"Error: column '{y}' is not in {self.df.columns.tolist()}"
    
        # If someone accidentally passes x == y, treat it like "plot y"
        if x == y:
            x = None
    
        # Scatter needs x
        if plot_type == "scatter" and x is None:
            return "Error: scatter plots need both x and y columns."
    
        title_csv = self.csv_name or "current CSV"
    
        if x is None:
            ax = self.df[y].plot(kind="line")
            ax.set_title(f"{title_csv} | Line plot: {y} vs row index")
            plt.show()
            return f"Plotted {y} vs row index as a line plot."
    
        if x not in self.df.columns:
            return f"Error: column '{x}' is not in {self.df.columns.tolist()}"
    
        ax = self.df.plot(x=x, y=y, kind=plot_type)
        ax.set_title(f"{title_csv} | {plot_type.title()} plot: {y} vs {x}")
        plt.show()
        
        return f"Plotted {y} vs {x} as a {plot_type}."
    ##----------------------------Q4-----------------------------------
    def compute_correlation(self, col1: str, col2: str):
        """
        Compute the Pearson correlation between two columns in the loaded DataFrame.
        Returns the correlation coefficient and p-value.
        """
        from scipy.stats import pearsonr

        if self.df is None:
            return {"error": "No CSV loaded"}

        if col1 not in self.df.columns:
            return {"error": f"Column '{col1}' not found"}

        if col2 not in self.df.columns:
            return {"error": f"Column '{col2}' not found"}

        data = self.df[[col1, col2]].dropna()

        pearson_r, p_value = pearsonr(data[col1], data[col2])

        return {
            "col1": col1,
            "col2": col2,
            "pearson_r": round(float(pearson_r), 4),
            "p_value": round(float(p_value), 4)
        }
    

print("Class defined")

csv_manager = CsvManager(RESOURCES_DIR)

node_tools = {
    "list_csv_files": csv_manager.list_csv_files,
    "load_csv": csv_manager.load_csv,
    "get_columns": csv_manager.get_columns,
    "summarize_columns": csv_manager.summarize_columns,
    "describe_column": csv_manager.describe_column,
    "plot_data": csv_manager.plot_data,
    "compute_correlation": csv_manager.compute_correlation,
}

tools_schema = [
    {
        "type": "function",
        "function": {
            "name": "list_csv_files",
            "description": "List available CSV files in the resources/ folder.",
        },
    },
    {
        "type": "function",
        "function": {
            "name": "load_csv",
            "description": "Load a CSV file from the resources/ folder and make it the active dataset.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {
                        "type": "string",
                        "description": "CSV filename in resources/, e.g. 'bike_commute.csv'.",
                    }
                },
                "required": ["filename"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_columns",
            "description": "Get the column names of the currently loaded CSV.",
        },
    },
    {
        "type": "function",
        "function": {
            "name": "summarize_columns",
            "description": "Show basic summary statistics for columns (uses pandas.describe).",
            "parameters": {
                "type": "object",
                "properties": {
                    "columns": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional list of column names. If omitted, summarize all columns.",
                    }
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "describe_column",
            "description": "Show basic summary statistics for a single column (uses pandas.describe).",
            "parameters": {
                "type": "object",
                "properties": {
                    "column": {
                        "type": "string",
                        "description": "Column name to describe.",
                    }
                },
                "required": ["column"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "plot_data",
            "description": "Plot data from the active CSV. If only y is provided, plot y vs row index.",
            "parameters": {
                "type": "object",
                "properties": {
                    "y": {"type": "string", "description": "Column name for y-axis."},
                    "x": {"type": "string", "description": "Optional column name for x-axis."},
                    "plot_type": {
                        "type": "string",
                        "enum": ["scatter", "line"],
                        "description": "Type of plot to create.",
                    },
                },
                "required": ["y"],
            },
        },
    },
    {
        
        "type": "function",
        "function": {
            "name": "compute_correlation",
            "description": (
                "Compute the Pearson correlation coefficient and p-value "
                "between two numeric columns in the currently loaded CSV."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "col1": {
                        "type": "string",
                        "description": "Name of the first numeric column."
                    },
                    "col2": {
                        "type": "string",
                        "description": "Name of the second numeric column."
                    }
                },
                "required": ["col1", "col2"]
            }
        }        
    },
    
]

def run_agent_cycle(messages, user_text, max_tool_rounds=5):
    """
    Run through one react-agent loop using a simple tool-using agent.
    `messages` parameter will usually just contain a system prompt, 
    and then user text will be appended.  

    The loop has three main steps:

    REASON:
      - Call the model with the conversation so far.
      - The model either replies normally, or asks to call a tool from tool set.

    ACT:
      - If tools are requested, run the Python functions

    OBSERVE:
      - Append each requested tool result back into the LLMs conversation history.
      - On the next iteration, the model reads those tool call results and determines
        whether it has reached the goal.

    Stop condition:
      - If the model returns an assistant message with no tool calls, this is the 
        final answer for this react cycle, this implies that reasoning alone without 
        tool calls was enough.  
      - max_tool_rounds is a safety cap to prevent infinite loops.
    """
    messages.append({"role": "user", "content": user_text})

    def observe_tool_result(tool_call_id, result):
        """
        Return a tool's return value as a message that can be appended to the
        LLMs conversation history. The model will read this tool output on the next
        REASON step.
        """
        content = json.dumps(result, default=str) if not isinstance(result, str) else result
        tool_message = {"role": "tool",
                        "tool_call_id": tool_call_id,
                        "content": content,}
        return tool_message

    for loop_idx in range(max_tool_rounds):
        # REASON: call the model
        # Here it will make use of any previous tool outputs it appended ("observed")
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=messages,
            tools=tools_schema,
        )

        msg = response.choices[0].message

        # Append the assistant message to the conversation history.
        # Use a plain dict so `messages` stays simple and inspectable.
        assistant_entry = {"role": "assistant", "content": msg.content}
        if msg.tool_calls:
            assistant_entry["tool_calls"] = [tc.model_dump() for tc in msg.tool_calls]
        messages.append(assistant_entry)

        # No tool calls means the model is answering directly.
        if not msg.tool_calls:
            return msg.content 

        # ACT + OBSERVE: run each tool call, then append its result.
        # Note there may be multiple tool calls
        for tool_call in msg.tool_calls:
            name = tool_call.function.name
            tool_args = json.loads(tool_call.function.arguments or "{}")

            print(f"ACT: {name}({tool_args})")

            fn = node_tools.get(name)
            if fn is None:
                result = {"error": f"Tool '{name}' not found."}
            else:
                try:
                    result = fn(**tool_args) if tool_args else fn()
                except Exception as e:
                    print(f"Tool error in {name}: {type(e).__name__}: {e}")
                    result = {"error": f"Tool '{name}' failed: {type(e).__name__}: {e}"}
                    
            # OBSERVE: append the tool result back into the conversation history.
            messages.append(observe_tool_result(tool_call.id, result))
            
            # After we appending information about all tool outputs, we loop back and REASON again.

    return "I hit the tool-round limit. Try a simpler request."

SYSTEM_PROMPT = (
    "You are a small data assistant for CSV files stored in resources/. "
    "Use the available tools to do any data work (do not guess). "
    "If no CSV is loaded yet, load one first (or list available CSV files). "
    "Keep answers short and student-friendly."
)

##------------------------------------------------Q5---------------------------------------------------

print("\n Question 5")
messages = [{"role": "system", "content": SYSTEM_PROMPT}]
result = run_agent_cycle(messages, "Load bike_commute.csv and compute the correlation between avg_traffic_density and avg_speed_kmh.")
print(result)

""" 
Agent's final response

The correlation between avg_traffic_density and avg_speed_kmh is approximately -0.532. This indicates a moderate negative
correlation, meaning that as traffic density increases, average speed tends to decrease. The p-value is 0.0, showing this 
result is statistically significant.

"""

















##---------------------------------------------------Question 6-----------------------------------------------------
print("\n Question 6")
##system = instructions, user = question, assistant = agent, tool = tool output
import json
print(json.dumps(messages, indent=2, default=str))




###----------------------------------------Lesson 04: smolagents---------------------------------------------


##------------------------------------------------Q7--------------------------------------------------------



@tool
def compute_correlation(col1: str, col2: str) -> dict:
    """
    Compute the Pearson correlation between two columns in the loaded CSV.

    Args:
        col1: Name of the first column.
        col2: Name of the second column.

    Returns:
        A dictionary containing the Pearson correlation and p-value.
    """
    return csv_manager.compute_correlation(col1, col2)

print("\n Question 7")
print(compute_correlation.description)

"""
smolagents automatically generates the tool description from the function name, type hints, and docstring and JSON schema
We don't need to write it manually. In Q4 we had to  manually write the JSON schema. Smolagents needs the developer to provide
a clear function name, parameter type hints, and descriptions for each parameter so it understand what the tool does and how to use it.

"""


##--------------------------------------------------Q8--------------------------------------------------------

@tool
def load_csv(filename: str) -> dict:
    """
    Load a CSV file from resources/ and make it the active dataset.

    Args:
        filename: The name of the CSV file to load.

    Returns:
        Information about the loaded CSV.
    """
    return csv_manager.load_csv(filename)

@tool
def plot_data(y: str, x: str, plot_type: str = "line") -> str:
    """
    Plot two columns from the loaded CSV and save the plot to a PNG file.

    Args:
        y: Column name for the y-axis.
        x: Column name for the x-axis.
        plot_type: Type of plot, such as scatter or line.

    Returns:
        A message describing the plot and where it was saved.
    """
    if csv_manager.df is None:
        return "Error: No CSV loaded."

    if x not in csv_manager.df.columns:
        return f"Error: column '{x}' not found."

    if y not in csv_manager.df.columns:
        return f"Error: column '{y}' not found."

    if plot_type not in ["scatter", "line"]:
        return "Error: plot type must be 'scatter' or 'line'."

    ax = csv_manager.df.plot(
        x=x,
        y=y,
        kind=plot_type,
        color="green"
    )

    ax.set_title(
        f"{csv_manager.csv_name} | "
        f"{plot_type.title()} plot: {y} vs {x}"
    )

    output_path = "assignments_07/outputs/q8_plot.png"

    plt.savefig(output_path)
    plt.close()

    return (
        f"Plotted {y} vs {x} as a {plot_type} with green dots "
        f"and saved it to {output_path}."
    )



TOOLS = [
    load_csv,
    plot_data,
    compute_correlation,
]

print("\n Question 8")
model_to_use = "gpt-4o-mini"  # default model ID
model = OpenAIServerModel(
    api_key=api_key,
    model_id=model_to_use,
)

tool_agent = ToolCallingAgent(
    tools=TOOLS,
    model=model
)

code_agent = CodeAgent(
    tools=TOOLS,
    model=model
)

prompt = "Load bike_commute.csv. Plot avg_heart_rate vs duration_min as a scatter plot with green dots."
print("\n Tool agent start ")
response_tool = tool_agent.run(prompt)
print("\n code agent start ")
response_code = code_agent.run(
    prompt,
    additional_args={"csv_manager": csv_manager}
)

print("\nToolCallingAgent response:")
print(response_tool)

print("\nCodeAgent response:")
print(response_code)



"""  

   Both agents were able to create the scatter plot of avg_heart_rate vs duration_min.
   The ToolCallingAgent called the load_csv and plot_data tools and the tool created
   the scatter plot with green dots.

   The CodeAgent also created the same scatter plot. It wrote Python code to call
   load_csv and plot_data.

   I first thought that the ToolCallingAgent might change the dot color because the
   prompt specifically asked for green dots. But looking at the result, the color is
   actually controlled by the plot_data function, not by the agent itself. So both
   agents were basically using the functionality that we gave them.
   
   
   
   I think the ToolCallingAgent is more useful when we already have specific tools
   that can do the task. It can decide which tool to call, but it is limited to the
   tools that we provide.

   The CodeAgent seems more useful when the task needs more flexible Python code.
   It can write and execute code, so it has more freedom to perform calculations,
   manipulate data, or combine different operations.


"""


###---------------------------------------------------Q9-----------------------------------------------------

"""
1. I think a ToolCallingAgent would be better for a task like checking the weather
   or getting the current time. The reason is that we can create a specific tool
   that does exactly what we need, and the agent just decides when to use that tool.

   This is a good fit because the task has a clear and limited set of actions.
   We don't need the agent to write and execute its own Python code. Using a
   predefined tool also makes the behavior more controlled and predictable.


2. One risk with a CodeAgent is that the agent actually generates and runs code.
   If the generated code is wrong or does something we did not expect, it could
   potentially modify files, delete data, or perform other unwanted actions,
   depending on what access the agent has.

   A ToolCallingAgent is more limited because it can only call the tools that we
   give it. So I think the main difference is that CodeAgent has more flexibility,
   but that flexibility also creates more risk.
"""