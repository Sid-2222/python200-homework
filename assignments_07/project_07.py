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
DATA_PATH = "assignments_01/outputs/merged_happiness.csv"



##----------------------------------------------------Task 1: Define Your Tools--------------------------------------------------

df = None

@tool
def load_happiness_data() -> dict:
    """Load the World Happiness dataset into memory.

    The function first attempts to load the merged happiness dataset from
    DATA_PATH. If the merged file does not exist, it loads all yearly CSV
    files from the happiness project resources directory, adds a ``Year``
    column based on each filename, and combines them into one DataFrame.

    Returns:
        dict: A dictionary containing the shape of the loaded DataFrame and
            a list of its column names.

    Raises:
        FileNotFoundError: If the fallback data directory does not exist or
            contains no CSV files.
        ValueError: If a yearly CSV filename does not contain a four-digit
            year.
    """
    global df

    if os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH)
    else:
        data_dir = "assignments/resources/happiness_project/"
        yearly_files = []

        if not os.path.exists(data_dir):
            raise FileNotFoundError(
                f"Data directory not found: {data_dir}"
            )

        for filename in os.listdir(data_dir):
            if filename.endswith(".csv"):
                yearly_files.append(os.path.join(data_dir, filename))

        yearly_files.sort()

        dataframes = []

        for filepath in yearly_files:
            yearly_df = pd.read_csv(filepath)

            # Extract the four-digit year from the filename.
            year_matches = re.findall(r"\b(19|20)\d{2}\b", os.path.basename(filepath))

            if not year_matches:
                raise ValueError(
                    f"Could not determine year from filename: {filepath}"
                )

            year = re.search(r"\b(?:19|20)\d{2}\b", os.path.basename(filepath)).group()

            yearly_df["Year"] = int(year)
            dataframes.append(yearly_df)

        if not dataframes:
            raise FileNotFoundError(
                f"No yearly CSV files found in {data_dir}"
            )

        df = pd.concat(dataframes, ignore_index=True)

    return {
        "shape": df.shape,
        "columns": df.columns.tolist()
    }
    
"""
Tool 2: summarize_column
"""

@tool
def summarize_column(column: str) -> dict:
    """Return descriptive statistics for a single column in the loaded dataset.

    Use this tool when you need summary statistics for one column, such
    as its count, mean, standard deviation, minimum, quartiles, or maximum.

    Args:
        column: The exact name of the column to summarize.

    Returns:
        dict: Descriptive statistics for the requested column as returned
        by pandas ``describe().to_dict()``. If no data is loaded or the
        column does not exist, returns a dictionary containing an error
        message.
    """
    if df is None:
        return {"error": "No data is loaded. Call load_happiness_data first."}

    if column not in df.columns:
        return {"error": f"Column '{column}' not found in the dataset."}

    return df[column].describe().to_dict() 



""" 
Tool 3: compute_correlation
"""

@tool
def compute_correlation(col1: str, col2: str) -> dict:
    """Compute the Pearson correlation coefficient and p-value between two numeric columns.

    Use this tool to measure the strength and direction of the linear
    relationship between two numeric columns in the loaded dataset.

    Args:
        col1: The exact name of the first numeric column.
        col2: The exact name of the second numeric column.

    Returns:
        dict: A dictionary containing:
            - col1: The name of the first column.
            - col2: The name of the second column.
            - pearson_r: Pearson correlation coefficient rounded to
              4 decimal places.
            - p_value: Two-sided Pearson correlation p-value rounded
              to 4 decimal places.

        If the data is not loaded, either column is missing, or the
        columns cannot be used for correlation, returns an error dictionary.
    """
    if df is None:
        return {"error": "No data is loaded. Call load_happiness_data first."}

    if col1 not in df.columns:
        return {"error": f"Column '{col1}' not found in the dataset."}

    if col2 not in df.columns:
        return {"error": f"Column '{col2}' not found in the dataset."}

    if not pd.api.types.is_numeric_dtype(df[col1]):
        return {"error": f"Column '{col1}' must be numeric."}

    if not pd.api.types.is_numeric_dtype(df[col2]):
        return {"error": f"Column '{col2}' must be numeric."}

    data = df[[col1, col2]].dropna()

    if len(data) < 2:
        return {"error": "Insufficient valid data to compute correlation."}

    if data[col1].nunique() < 2 or data[col2].nunique() < 2:
        return {"error": "Correlation cannot be computed for a constant column."}

    try:
        r, p_value = pearsonr(data[col1], data[col2])
    except (ValueError, TypeError) as exc:
        return {"error": f"Could not compute correlation: {exc}"}

    return {
        "col1": col1,
        "col2": col2,
        "pearson_r": round(r, 4),
        "p_value": round(p_value, 4),
    }
    
    
"""
Tool 4: get_top_n_countries
"""

@tool
def get_top_n_countries(column: str, year: int, n: int = 5) -> dict:
    """Return the top N countries ranked by a given column for a specific year.

    Filters the loaded dataset to the requested year, sorts the rows by
    the specified column in descending order, and returns the top N
    countries and their corresponding values.

    Args:
        column: The exact name of the column used for ranking.
        year: The year to filter the dataset by.
        n: The maximum number of countries to return. Defaults to 5.

    Returns:
        dict: A dictionary containing a ``results`` key. The value is a
        list of dictionaries, where each dictionary contains ``country``
        and the requested column value.

        If the data is not loaded, the column or year is unavailable, or
        n is invalid, returns a dictionary containing an error message.
    """
    if df is None:
        return {"error": "No data is loaded. Call load_happiness_data first."}

    if column not in df.columns:
        return {"error": f"Column '{column}' not found in the dataset."}

    if "Year" not in df.columns:
        return {"error": "Column 'Year' not found in the dataset."}

    if "Country" not in df.columns:
        return {"error": "Column 'Country' not found in the dataset."}

    if not isinstance(n, int) or isinstance(n, bool) or n <= 0:
        return {"error": "n must be a positive integer."}

    try:
        year = int(year)
    except (TypeError, ValueError):
        return {"error": "year must be an integer."}

    year_df = df[df["Year"] == year]

    if year_df.empty:
        return {"error": f"No data found for year {year}."}

    try:
        top_rows = (
            year_df[["Country", column]]
            .dropna(subset=[column])
            .sort_values(by=column, ascending=False)
            .head(n)
        )
    except (TypeError, ValueError) as exc:
        return {"error": f"Could not sort by column '{column}': {exc}"}

    results = [
        {
            "country": row["Country"],
            column: row[column],
        }
        for _, row in top_rows.iterrows()
    ]

    return {"results": results}


 
##--------------------------------------------Task 2: Build the Agent-------------------------------------------------------------------------------



model = OpenAIServerModel(api_key=api_key, model_id="gpt-4o-mini")

SYSTEM_PROMPT = """
You are a data analyst assistant for the World Happiness dataset.

Use the available tools for loading data, summarizing columns, computing
correlations, and ranking countries. Write Python code directly only when
the tools are not sufficient, such as when creating custom plots.

Important:
- load_happiness_data() loads the dataset and returns a dictionary containing
  only the dataset shape and column names.
- Do not treat the return value of load_happiness_data() as a pandas DataFrame.
- The global DataFrame inside the tool functions is not directly available
  to CodeAgent-generated Python code.
- When custom analysis or plotting requires the full dataset, read the CSV
  directly using pandas from:
  assignments_01/outputs/merged_happiness.csv

The exact dataset column names are:
"Ranking"
"Country"
"Year"
"Regional indicator"
"Happiness score"
"GDP per capita"
"Social support"
"Healthy life expectancy"
"Freedom to make life choices"
"Generosity"
"Perceptions of corruption"

Always use these exact column names. Do not substitute snake_case names such
as happiness_score or gdp_per_capita.

For custom plots:
- Use pandas and matplotlib.
- Use "Regional indicator" for region information.
- Use "Year" for the x-axis.
- Use "Happiness score" for happiness values.
- Save plots to the exact path requested by the user.
- Create the output directory if necessary.

Be concise and student-friendly in your responses.
"""

agent = CodeAgent(
    tools=[load_happiness_data, summarize_column, compute_correlation, get_top_n_countries],
    model=model,
    instructions=SYSTEM_PROMPT,
    additional_authorized_imports=["pandas", "matplotlib.pyplot", "scipy.stats"],
    max_steps=8,
)


 
##--------------------------------------------Task 3: Run Guided Queries----------------------------------------------------------------------

if __name__ == "__main__":
    
    # queries = [
    #     "Load the happiness data and tell me its shape and column names.",
    #     "Summarize the Happiness score column.",
    #     "What is the correlation between GDP per capita and Happiness score? Is it statistically significant?",
    #     "Show me the top 5 happiest countries in 2020.",
    #     "Plot Happiness score over the years as a line chart, with one line per region. Save the plot to assignments_07/outputs/happiness_by_region.png.",
    # ]
    
    queries = [
        "Load the happiness data and tell me its shape and column names.",
        "Summarize the Happiness score column.",
        "What is the correlation between GDP per capita and Happiness score? Is it statistically significant?",
        "Show me the top 5 happiest countries in 2020.",
        "Plot Happiness score over the years as a line chart, with one line per Regional indicator. Save the plot to outputs/happiness_by_region.png.",
        ]

    for query in queries:
        print(f"\n--- Query: {query} ---")
        response = agent.run(query, reset=False)
        print(response)
    
    
##--------------------------------------------Task 4: Your Own Questions---------------------------------------------------------
        
    # My query 1
    my_query_1 = "What is the correlation between Healthy life expectancy and GDP per capita?"

    response_1 = agent.run(my_query_1, reset=False)
    print(f"\n--- Query: {my_query_1} ---")
    print(response_1)
    """ 
        Comment: This query triggered the compute_correlation tool:
        compute_correlation("Healthy life expectancy", "GDP per capita")
        It did not require the agent to generate custom Python code.


    """


    # My query 2
    my_query_2 = "Plot a line chart for the United States using the columns Happiness score, GDP per capita, Social support, Healthy life expectancy, Freedom to make life choices, Generosity, and Perceptions of corruption for all available years. Save the plot to outputs/usa_line_chart.png."

    response_2 = agent.run(my_query_2, reset=False)
    print(f"\n--- Query: {my_query_2} ---")
    print(response_2)

    """ 
        Comment: This query did not require any of the available tools.
        The agent generated Python code using pandas and matplotlib to create
        the requested line chart and save it to the specified output path.
    """





    ##------------------------------------------Task 5: Reflection---------------------------------------------------------------------------

    """ 
    --- Reflection ---

    1. In Query 3, how did the agent communicate whether the correlation was statistically
    significant? Did it use the p-value correctly? What threshold did it apply?
    
    A: The agent communicated the correlation was statistically significant using the 
        p_value (0.0) which is less than the threshold < 0.05.
        
        It usedd the p_value correctly.
        
        It used the threshold < 0.05

    2. Did any of the agent's responses surprise you — either by being more capable than
    you expected, or less? Describe one specific example.
    
    A: Yes, the agent's response did surprise me, sometimes it runs into some error and check it and corrects itself.
        and this seems so powerful to me.

    3. What one additional tool would make this agent meaningfully more useful?
    Describe what it would do and what kind of question it would help the agent answer.
    (You do not need to implement it.)
    
    A: I think a data visualization tool would make the agent more useful. It could create 
        charts like line graphs, bar charts, and scatter plots. This would help answer questions like,
        “How has India's happiness score changed over the years?” or “Is GDP related to happiness?”
        It would make the results easier to understand and more visual.

    """