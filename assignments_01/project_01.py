from prefect import task, flow, get_run_logger
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import pearsonr


pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200) 
pd.set_option('display.max_colwidth', None)

files = [
    "https://raw.githubusercontent.com/Code-the-Dream-School/python-200-v1/main/assignments/resources/happiness_project/world_happiness_2015.csv",
    "https://raw.githubusercontent.com/Code-the-Dream-School/python-200-v1/main/assignments/resources/happiness_project/world_happiness_2016.csv",
    "https://raw.githubusercontent.com/Code-the-Dream-School/python-200-v1/main/assignments/resources/happiness_project/world_happiness_2017.csv",
    "https://raw.githubusercontent.com/Code-the-Dream-School/python-200-v1/main/assignments/resources/happiness_project/world_happiness_2018.csv",
    "https://raw.githubusercontent.com/Code-the-Dream-School/python-200-v1/main/assignments/resources/happiness_project/world_happiness_2019.csv",
    "https://raw.githubusercontent.com/Code-the-Dream-School/python-200-v1/main/assignments/resources/happiness_project/world_happiness_2020.csv",
    "https://raw.githubusercontent.com/Code-the-Dream-School/python-200-v1/main/assignments/resources/happiness_project/world_happiness_2021.csv",
    "https://raw.githubusercontent.com/Code-the-Dream-School/python-200-v1/main/assignments/resources/happiness_project/world_happiness_2022.csv",
    "https://raw.githubusercontent.com/Code-the-Dream-School/python-200-v1/main/assignments/resources/happiness_project/world_happiness_2023.csv",
    "https://raw.githubusercontent.com/Code-the-Dream-School/python-200-v1/main/assignments/resources/happiness_project/world_happiness_2024.csv",
]


@task(retries=3 , retry_delay_seconds=2)
def load_data(files):
    logger = get_run_logger()
    data_frame = []   
    
    for file in files:
        year = int(file.split("world_happiness_")[1].replace(".csv", ""))
        df = pd.read_csv(file ,  sep=";",  decimal=",")
        if "Ladder score" in df.columns:
            df.rename(columns={"Ladder score": "Happiness score"}, inplace=True)
        country_position = df.columns.get_loc("Country")
        df.insert(country_position + 1, "Year", year)
        data_frame.append(df)
    logger.info(f"Loaded {len(data_frame)} yearly happiness datasets")
    return data_frame

@task
def merge_data(data_frame):
    logger = get_run_logger()
    merged_df = pd.concat(data_frame, ignore_index=True)
    merged_df =  merged_df.sort_values(by=["Country", "Year"]).reset_index(drop=True)
    logger.info(f"Merged dataset contains {len(merged_df)} rows")
    return merged_df

@task(retries=3, retry_delay_seconds=2)
def save_data(merged_df):
    logger = get_run_logger()
    merged_df.to_csv("outputs/merged_happiness.csv", index=False)
    logger.info("merged_happiness.csv saved successfully")

# Task 2: Descriptive Statistics
@task
def analyze_data(merged_df):
    logger = get_run_logger()
    mean_score = merged_df["Happiness score"].mean()
    median_score = merged_df["Happiness score"].median()
    sd_score = merged_df["Happiness score"].std()
    
    logger.info(f"Mean happiness score: {mean_score:.2f}")
    logger.info(f"median happiness score: {median_score:.2f}")
    logger.info(f"Standard deviation: {sd_score:.2f}")
    
    yearly_mean = merged_df.groupby("Year")["Happiness score"].mean()
    logger.info(f"Mean happiness score by year: {yearly_mean}")
    
    region_mean = merged_df.groupby("Regional indicator")["Happiness score"].mean().sort_values(ascending=False)
    logger.info(f"Mean happiness score by Region: {region_mean}")
    
    return {
        "mean": mean_score,
        "median": median_score,
        "std": sd_score,
        "yearly_mean": yearly_mean,
        "regional_mean": region_mean
    }
    
@task
def make_histogram(merged_df):
    logger = get_run_logger()
    plt.hist(merged_df["Happiness score"], bins= 20, color= "blue", rwidth=0.85)
    plt.title("Histogram of Happiness Scores (2015–2024)")
    plt.xlabel("Happiness Score")
    plt.ylabel("Number of Countries") 
    plt.savefig("outputs/happiness_histogram.png", bbox_inches="tight")
    plt.show()
    plt.close()
    logger.info("Histogram saved successfully: outputs/happiness_histogram.png")

@task
def make_boxplot(merged_df):
    logger = get_run_logger()
    years = sorted(merged_df["Year"].unique())
    data = [
    merged_df[merged_df["Year"] == year]["Happiness score"]
    for year in years
    ]
    plt.boxplot(data, labels=years)
    plt.title("Happiness Score Distribution by Year")
    plt.xlabel("Year")
    plt.ylabel("Happiness Score")    
    plt.savefig("outputs/happiness_by_year.png", bbox_inches="tight")
    plt.show()
    plt.close()
    logger.info("Boxplot saved successfully: outputs/happiness_by_year.png")
    
@task
def make_scatter(merged_df):
    logger = get_run_logger()
    plt.scatter(merged_df["GDP per capita"], merged_df["Happiness score"])
    plt.title("GDP vs Happiness")
    plt.xlabel("GDP")
    plt.ylabel("Happiness")    
    plt.savefig("outputs/gdp_vs_happiness.png", bbox_inches="tight")
    plt.show()
    plt.close()
    logger.info("Boxplot saved successfully: outputs/gdp_vs_happiness.png")
    
@task
def make_heatmap(merged_df):
    logger = get_run_logger()
    numeric_df = merged_df.select_dtypes(include="number")
    correlation_matrix = numeric_df.corr(method="pearson")
    
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("correlation heatmap")
    plt.savefig("outputs/correlation_heatmap.png", bbox_inches="tight")
    plt.show()
    plt.close()
    logger.info("Boxplot saved successfully: outputs/correlation_heatmap.png")
    
@task
def pandemic_ttest(merged_df):
    logger = get_run_logger()
    scores_2019 = merged_df.loc[merged_df["Year"] == 2019, "Happiness score"].dropna()
    scores_2020 = merged_df.loc[merged_df["Year"] == 2020, "Happiness score"].dropna()
    mean_2019 = scores_2019.mean()
    mean_2020 = scores_2020.mean()
    t_stat, p_val = stats.ttest_ind(scores_2019, scores_2020, equal_var=False)
    logger.info(f"2019 mean happiness score: {mean_2019:.3f}")
    logger.info(f"2020 mean happiness score: {mean_2020:.3f}")
    logger.info(f"T-statistic: {t_stat:.8f}")
    logger.info(f"P-value: {p_val}")
    alpha = 0.05
    
    if p_val < alpha:
        interpretation = (
            "There is a statistically significant difference in global "
            "happiness scores between 2019 and 2020. The data suggests that "
            "the pandemic period was associated with a meaningful change "
            "in reported happiness levels."
        )
    else:
        interpretation = (
            "There is no statistically significant difference in global "
            "happiness scores between 2019 and 2020. Based on this dataset, "
            "we do not have enough evidence to conclude that the pandemic "
            "caused a measurable change in global happiness scores."
        )

    logger.info(f"Interpretation: {interpretation}")

    return {
        "2019_mean": mean_2019,
        "2020_mean": mean_2020,
        "t_statistic": t_stat,
        "p_value": p_val,
        "interpretation": interpretation
    }
    
@task
def region_ttest(merged_df):
    logger = get_run_logger()
    western_europe = merged_df.loc[merged_df["Regional indicator"] == "Western Europe","Happiness score"].dropna()
    sub_saharan_africa = merged_df.loc[merged_df["Regional indicator"] == "Sub-Saharan Africa","Happiness score"].dropna()
    mean_western = western_europe.mean()
    mean_africa = sub_saharan_africa.mean()
    t_stat, p_value = stats.ttest_ind(western_europe, sub_saharan_africa, equal_var=False)
    
    logger.info(f"Western Europe mean happiness score: {mean_western:.3f}")
    logger.info(f"Sub-Saharan Africa mean happiness score: {mean_africa:.3f}")
    logger.info(f"Regional t-statistic: {t_stat:.8f}")
    logger.info(f"Regional p-value: {p_value}")
    
    if p_value < 0.05:
        interpretation = (
            "Western Europe and Sub-Saharan Africa have a statistically "
            "significant difference in happiness scores. The difference "
            "observed in the descriptive statistics is unlikely to be "
            "explained by random variation alone."
        )
    else:
        interpretation = (
            "The difference between Western Europe and Sub-Saharan Africa "
            "is not statistically significant in this dataset."
        )

    logger.info(f"Regional interpretation: {interpretation}")

    return {
        "western_europe_mean": mean_western,
        "sub_saharan_africa_mean": mean_africa,
        "t_statistic": t_stat,
        "p_value": p_value,
        "interpretation": interpretation
    }


@task
def correlation_analysis(merged_df):
    logger = get_run_logger()
    alpha = 0.05
    numeric_columns = merged_df.select_dtypes(include="number").columns
    explanatory_variables = [
        col for col in numeric_columns 
        if col != "Happiness score"
    ]

    results = []
    for variable in explanatory_variables:
        temp_df = merged_df[[variable, "Happiness score"]].dropna()
        correlation, p_value = pearsonr(temp_df[variable],temp_df["Happiness score"])
        significant_original = p_value < alpha
        results.append({
            "variable": variable,
            "correlation": correlation,
            "p_value": p_value,
            "significant_alpha_0.05": significant_original
        })
        logger.info(
                f"{variable}: "
                f"correlation={correlation:.8f}, "
                f"p-value={p_value}, "
                f"significant at alpha=0.05: {significant_original}"
            )
    number_of_tests = len(results)
    adjusted_alpha = alpha / number_of_tests
    logger.info(f"Number of correlation tests performed: {number_of_tests}")

    logger.info(f"Bonferroni adjusted alpha: {adjusted_alpha:.10f}")
    for result in results:
        result["significant_after_bonferroni"] = (
            result["p_value"] < adjusted_alpha
        )

        logger.info(
            f"{result['variable']} remains significant after "
            f"Bonferroni correction: "
            f"{result['significant_after_bonferroni']}"
        )

    return results
    
@task
def make_report(merged_data, pandemic_result,correlation_result):
    logger = get_run_logger()
    number_of_country = merged_data["Country"].nunique()
    total_years = merged_data["Year"].nunique()
    regiona_mean = merged_data.groupby("Regional indicator")["Happiness score"].mean().sort_values(ascending=False)
    
    top_3 = regiona_mean.head(3)
    bottom_3 = regiona_mean.tail(3)
    
    significant = [
        r for r in correlation_result
        if r["significant_after_bonferroni"]
    ]

    strongest = max(significant, key=lambda x: abs(x["correlation"]))
    
    logger.info("========== The Happiness Pipeline Report ==========")
    logger.info(f"The Dataset contains {number_of_country} Countries and have {total_years} years of data")
    
    logger.info("Top 3 happiest regions by average")
    for region , score in top_3.items():
       logger.info(f" {region} -> {score}") 
    
    logger.info("Bottom 3 happiest regions by average")
    for region , score in bottom_3.items():
       logger.info(f" {region} -> {score}")

    if pandemic_result["p_value"] < 0.05:
        interpretation = (
            "There was a statistically significant difference in global "
            "happiness scores between 2019 and 2020."
        )
    else:
        interpretation = (
            "There was no statistically significant difference in global "
            "happiness scores between 2019 and 2020."
        )
    logger.info(interpretation)

    logger.info(
        f"The strongest variable associated with happiness score after "
        f"Bonferroni correction was '{strongest['variable']}' "
        f"(r = {strongest['correlation']:.5f}, "
        f"p = {strongest['p_value']:.10g})."
    )

    logger.info("========== End Report ==========")
    


    


    
@flow
def happiness_pipeline():
    dataframe = load_data(files)
    merged_data = merge_data(dataframe)
    analyze_data(merged_data)
    save_data(merged_data)
    make_histogram(merged_data)
    make_boxplot(merged_data)
    make_scatter(merged_data)
    make_heatmap(merged_data)
    pandemic_result = pandemic_ttest(merged_data)
    region_ttest(merged_data)
    correlation_result = correlation_analysis(merged_data)
    make_report(merged_data, pandemic_result, correlation_result)

    return merged_data
    




if __name__ == "__main__":
    happiness_pipeline()
    

    
    
    
    