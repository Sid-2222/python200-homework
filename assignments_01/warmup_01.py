import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statistics as sat
from scipy import stats
from scipy.stats import pearsonr
import seaborn as sns

 # --- Pandas ---
 
# Pandas Q1)
 
data = {
    "name":   ["Alice", "Bob", "Carol", "David", "Eve"],
    "grade":  [85, 72, 90, 68, 95],
    "city":   ["Boston", "Austin", "Boston", "Denver", "Austin"],
    "passed": [True, True, True, False, True]
}
df = pd.DataFrame(data)
print(df.head(3))
print("Shape:", df.shape)
print("Data Type:", df.dtypes)

# Pandas Q2)
grade_above_80 = df[ (df["passed"] == True ) &  (df["grade"] > 80)]
print(grade_above_80)

# Pandas Q3)

df["grade_curved"] = df["grade"] + 5
print(df)

# Pandas Q4)

df["name_upper"] = df["name"].str.upper()
print(df[["name", "name_upper"]])

# Pandas Q5)
by_city = df.groupby(["city"])["grade"].mean()
print(by_city)

# Pandas Q6)
df["city"] = df["city"].replace("Austin", "Houston")
print(df[["name", "city"]])

# Pandas Q7)

grade_sort = df.sort_values(by= "grade" , ascending= False)
print(grade_sort.head(3))

 # --- NumPy ---
# NumPy Question 1

array = np.array([10, 20, 30, 40, 50])
print("Shape:", array.shape)
print("Data Type:", array.dtype)
print("Number of Dimensions:", array.ndim)

# NumPy Question 2)

import numpy as np
arr = np.array([[1, 2, 3],
                [4, 5, 6],
                [7, 8, 9]])

print("Shape:", arr.shape)
print("Size:", arr.size)

# NumPy Question 3)
sliced_array = arr[0:2 , 0:2]
print(sliced_array)

# NumPy Question 4)

q4_array = np.zeros((3,4))
q4_array2 = np.ones((2,5))
print(q4_array)
print(q4_array2)

# NumPy Question 5)

q5_array = np.arange(0, 50, 5)
print("question 5")
print(q5_array)
print("Shape:", q5_array.shape)
print("Mean:" , q5_array.mean())
print("Sum:" , q5_array.sum())
print("standard deviation: " , q5_array.std())

# NumPy Question 6)

q6_array = np.random.normal(loc=0, scale=1, size=200) # loc is mean , scale is Standared deviation

print("question 6")
print("Mean:" , q6_array.mean())
print("standard deviation: " , q6_array.std())

 # --- Matplotlib ---
# Matplotlib Question 1)

x = [0, 1, 2, 3, 4, 5]
y = [0, 1, 4, 9, 16, 25]

plt.plot(x,y, marker='o')
plt.title("Squares")
plt.xlabel("x")
plt.ylabel("y")
plt.show()

# Matplotlib Question 2)

subjects = ["Math", "Science", "English", "History"]
scores   = [88, 92, 75, 83]

plt.bar(subjects,scores)
plt.title("Subject Scores")
plt.xlabel("subjects")
plt.ylabel("score")
plt.show()

# Matplotlib Question 3)

x1, y1 = [1, 2, 3, 4, 5], [2, 4, 5, 4, 5]
x2, y2 = [1, 2, 3, 4, 5], [5, 4, 3, 2, 1]

plt.scatter(x1, y1, color = "blue",  marker= "*",  label="Dataset 1" )
plt.scatter(x2, y2, color = "red",  marker= "^",  label="Dataset 2" )
plt.title("2 Data sets on same figure")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend(loc="best")
plt.grid(True)
plt.show()

# Matplotlib Question 4)
fig, ax = plt.subplots(1, 2) # creating figure 1 row 2 column
ax[0].plot(x,y, marker= "*")
ax[0].set_title("Squares")
ax[0].set_xlabel("x")
ax[0].set_ylabel("y")


ax[1].bar(subjects,scores)
ax[1].set_title("subject scores")
ax[1].set_xlabel("subjects")
ax[1].set_ylabel("scores")

plt.tight_layout()
plt.show()

# --- Descriptive Stats ---

# Descriptive Stats Question 1)
data = [12, 15, 14, 10, 18, 22, 13, 16, 14, 15]
print("mean data :" , np.mean(data))
print("median data :" , np.median(data))
print("variance :" , np.var(data))
print("deviation :" , np.std(data))

# Descriptive Stats Question 2)
dsq2_arr = np.random.normal(65, 10, 500)

plt.hist(dsq2_arr, bins= 20, color= "blue", rwidth=0.85)
plt.title("Distribution of Scores")
plt.xlabel("Scores")
plt.ylabel("Frequency")
plt.show()

# Descriptive Stats Question 3)

group_a = [55, 60, 63, 70, 68, 62, 58, 65]
group_b = [75, 80, 78, 90, 85, 79, 82, 88]

plt.boxplot([group_a , group_b], labels=["Group A", "Group B"])
plt.title("Score compare")
plt.ylabel("Scores")
plt.show()

# Descriptive Stats Question 4)

normal_data = np.random.normal(50, 5, 200)
skewed_data = np.random.exponential(10, 200)
plt.boxplot([normal_data, skewed_data], tick_labels= ["Normal", "Exponential"])
plt.title("Distribution Comparison")
plt.ylabel("Values")
plt.show()

# the skewed data is more skewed with more outliers values 
# where the normal data have more central tendency with few outliers

# Descriptive Stats Question 5)

data1 = [10, 12, 12, 16, 18]
data2 = [10, 12, 12, 16, 150]
print("data 1 Mean :" , np.mean(data1))
print("data 1 median :" , np.median(data1))
print("data 1 mode :" , sat.mode(data1))

print("data 2 Mean :" , np.mean(data2))
print("data 2 median :" , np.median(data2))
print("data 2 mode :" , sat.mode(data2))

# The median and mean are so diffrent because of one value of data2 which is 150 which shoots up the mean value of data2 because it seems its a outliers

# --- DHypothesis ---
# Hypothesis Question 1)

group_a = [72, 68, 75, 70, 69, 73, 71, 74]
group_b = [80, 85, 78, 83, 82, 86, 79, 84]
alpha = 0.05
t_stat, p_value = stats.ttest_ind(group_a, group_b)

print("t-statistic :", t_stat)
print("p-value :", p_value)

# Hypothesis Question 2)

if(p_value >= 0.05):
    print(f"The result is not statistically significant because the p-value ({p_value:.8f}) is greater than alpha ({alpha}).")
else:
    print(f"The result is statistically significant because the p-value ({p_value:.8f}) is less than alpha ({alpha}).")
    
# Hypothesis Question 3)

before = [60, 65, 70, 58, 62, 67, 63, 66]
after  = [68, 70, 76, 65, 69, 72, 70, 71]

t_stat , p_val = stats.ttest_rel(before, after)

print("t-statistic :", t_stat)
print("p-value :", p_val)

# Hypothesis Question 4)

scores = [72, 68, 75, 70, 69, 74, 71, 73]

t_stat , p_val = stats.ttest_1samp(scores,70)
print("t-statistic :", t_stat)
print("p-value :", p_val)

# Hypothesis Question 5)

t_stat, p_value = stats.ttest_ind(group_a, group_b, alternative="less")

print("t-statistic :", t_stat)
print("p-value :", p_value)

# Hypothesis Question 6)
mean_a = np.mean(group_a)
mean_b = np.mean(group_b)

print(
    f"Group B had a higher average score ({mean_b:.3f}) than Group A ({mean_a:.3f}). "
    f"The difference is statistically significant (p-value = {p_value:.8f}), "
    "so it is unlikely that this difference happened by chance."
)

# --- Correlation ---

# Correlation Question 1)
x = [1, 2, 3, 4, 5]
y = [2, 4, 6, 8, 10]
corr = np.corrcoef(x, y)
print(corr)
print(corr[0, 1])

#from the output we can see a perfect positive corellation where y = x * 2

# ==========================
# Correlation Question 2
# Printing Correlation & p-value
# ==========================


x = [1,  2,  3,  4,  5,  6,  7,  8,  9, 10]
y = [10, 9,  7,  8,  6,  5,  3,  4,  2,  1]

corr_val ,p_val = pearsonr(x, y)
print("Correlation:", round(corr_val, 8))
print("p-value:", p_val)

# ==========================
# Correlation Question 3
# Create DataFrame and calculate correlation matrix
# ==========================

people = {
    "height": [160, 165, 170, 175, 180],
    "weight": [55,  60,  65,  72,  80],
    "age":    [25,  30,  22,  35,  28]
}
df = pd.DataFrame(people)

corr_df = df.corr()
print(corr_df)

# ==========================
# Correlation Question 4
# Create scatter plot showing negative relationship
# ==========================

import matplotlib.pyplot as plt
x = [10, 20, 30, 40, 50]
y = [90, 75, 60, 45, 30]

plt.scatter(x, y, color="teal")
plt.title("Negative Correlation")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()

# ==========================
# Correlation Question 5
# Create heatmap using the correlation matrix from Q3
# ==========================

sns.heatmap(corr_df, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()

# --- Pipeline ---
#Pipeline Question 1)
import pandas as pd
import numpy as np
arr = np.array([12.0, 15.0, np.nan, 14.0, 10.0, np.nan, 18.0, 14.0, 16.0, 22.0, np.nan, 13.0])

def create_series(arr):
    return pd.Series(arr, name= "values")

def clean_data(series):
    return series.dropna()

def summarize_data(series):
    return {
        "mean" : series.mean(),
        "median": series.median(),
        "std" : series.std(),
        "mode" : series.mode()[0]
    }
def data_pipeline(arr):
    series = create_series(arr)
    clear_series = clean_data(series)
    final_summary = summarize_data(clear_series)
    return final_summary

pipe = data_pipeline(arr)

for key, value in pipe.items():
    print(f"{key}: {value}")