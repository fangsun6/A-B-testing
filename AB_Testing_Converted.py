from IPython.display import display, HTML
video_id = 'GaP5f0jVTWE'
iframe_html = f'''
<iframe width="560" height="315" src="https://www.youtube.com/embed/{video_id}"
frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope;
picture-in-picture" allowfullscreen></iframe>
'''
display(HTML(iframe_html))
import numpy as np  
import pandas as pd  
import seaborn as sns  
import matplotlib.pyplot as plt  
import os  
from scipy.stats import shapiro, ttest_ind, mannwhitneyu, levene
import scipy.stats as stats  
import statsmodels.stats.api as sms
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")
warnings.simplefilter(action='ignore', category=FutureWarning)
pd.set_option('display.max_columns', None)  
pd.options.display.float_format = '{:.4f}'.format  
import gdown
import pandas as pd
file_id = '1IkN-fylT9ZYxZJhzbgpAmJ751b6lrPy9'
download_url = f'https://drive.google.com/uc?id={file_id}'
output = 'cookie_cats.csv'
gdown.download(download_url, output, quiet=False)
raw = pd.read_csv(output)
print(raw.head())
import gdown
import pandas as pd
import numpy as np
import io

def load_from_drive(file_id, file_type, info=True):
    download_url = f'https://drive.google.com/uc?id={file_id}'

    output = f'downloaded_file.{file_type}'

    gdown.download(download_url, output, quiet=False)

   
    if file_type == 'csv':
        read = pd.read_csv(output)  
    elif file_type == 'xlsx':
        read = pd.read_excel(output)  
    else:
        raise ValueError("Unsupported file type. Please use 'csv' or 'xlsx'.")

    if len(read) > 0:
    print("# Data has been imported!")
    print("# ------------------------------------", "\n")

    print("# Dimensions -------------------------")
    print("Number of observations:", read.shape[0], "Number of columns:", read.shape[1], "\n")

    print("# Data Types -----------------------------")
    if len(read.select_dtypes("object").columns) > 0:
        print("Object (string) variables:", "\n", "Number of variables:",
              len(read.select_dtypes("object").columns), "\n",
              read.select_dtypes("object").columns.tolist(), "\n")

    if len(read.select_dtypes("int64").columns) > 0:
        print("Integer variables:", "\n", "Number of variables:",
              len(read.select_dtypes("int64").columns), "\n",
              read.select_dtypes("int64").columns.tolist(), "\n")

    if len(read.select_dtypes("float64").columns) > 0:
        print("Floating-point variables:", "\n", "Number of variables:",
              len(read.select_dtypes("float64").columns), "\n",
              read.select_dtypes("float64").columns.tolist(), "\n")

    if len(read.select_dtypes("bool").columns) > 0:
        print("Boolean variables:", "\n", "Number of variables:",
              len(read.select_dtypes("bool").columns), "\n",
              read.select_dtypes("bool").columns.tolist(), "\n")

    # Print missing value info
    print("# Missing Values ---------------------")
    print("Any missing values? \n ", np.where(read.isnull().values.any() == False,
                                          "No missing values!", "Data contains missing values!"), "\n")

    # Print memory usage
    buf = io.StringIO()
    read.info(buf=buf)
    info = buf.getvalue().split('\n')[-2].split(":")[1].strip()
    print("# Memory Usage ---------------------- \n", info)

else:
    print("# Data has not been imported!")


    return read

file_id = '1IkN-fylT9ZYxZJhzbgpAmJ751b6lrPy9'  
file_type = 'csv' 
ab = load_from_drive(file_id, file_type)
ab.head()

ab["userid"].nunique()
# Counting the number of players in each A/B group.
ab.groupby("version")[["userid"]].nunique()
ab['version'].value_counts()
print(ab.userid.nunique() == ab.shape[0])  
ab.describe([0.01, 0.05, 0.10, 0.20, 0.80, 0.90, 0.95, 0.99])[["sum_gamerounds"]].T  
ab.groupby("version").sum_gamerounds.agg(["count", "median", "mean", "std", "max"])
get_ipython().system('wget -q -O TaipeiSansTCBeta-Regular.ttf https://drive.google.com/uc?id=1eGAsTN1HBpJAkeVM57_C7ccp7hbgSz3_&export=download')

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.font_manager import fontManager

fontManager.addfont('TaipeiSansTCBeta-Regular.ttf')
mpl.rc('font', family='Taipei Sans TC Beta')


fig, axes = plt.subplots(1, 3, figsize=(18, 5))

ab[(ab.version == "gate_30")].hist("sum_gamerounds", ax=axes[0], color="steelblue")


ab[(ab.version == "gate_40")].hist("sum_gamerounds", ax=axes[1], color="steelblue")


sns.boxplot(x=ab.version, y=ab.sum_gamerounds, ax=axes[2])


plt.suptitle("Before Removing Outliers", fontsize=20)
axes[0].set_title("Distribution of Gate 30 Group (A)", fontsize=15)
axes[1].set_title("Distribution of Gate 40 Group (B)", fontsize=15)
axes[2].set_title("Distribution of Both Groups", fontsize=15)



plt.tight_layout(pad=4)

# Calculating 1-day and 7-days retention for each A/B group
df_retention_ab = ab.groupby("version").agg({"userid":"count", "retention_1":"mean","retention_7":"mean", "sum_gamerounds":"sum"})
df_retention_ab


# Select data from the dataset where the version is "gate_30" and "gate_40",
# and plot the line chart of total game rounds played
ab[ab.version == "gate_30"].sum_gamerounds.plot(legend=True, label="Gate 30", figsize=(20, 5))
ab[ab.version == "gate_40"].sum_gamerounds.plot(legend=True, label="Gate 40")

# Set the title of the plot
plt.title("Before Removing the Extreme Value", fontsize=20)



# Keep only rows where game rounds are less than the maximum value
ab = ab[ab.sum_gamerounds < ab.sum_gamerounds.max()]

# Calculate summary statistics for game rounds
# Select specific percentiles and transpose the result for better readability
ab.describe([0.01, 0.05, 0.10, 0.20, 0.80, 0.90, 0.95, 0.99])[["sum_gamerounds"]].T


# In[16]:


# Create subplots with 1 row and 4 columns, set figure size to (18, 5)
fig, axes = plt.subplots(1, 4, figsize=(18, 5))

# Plot a histogram of total game rounds on the first subplot
ab.sum_gamerounds.hist(ax=axes[0], color="steelblue")

# Plot a histogram of game rounds for "gate_30" group on the second subplot
ab[(ab.version == "gate_30")].hist("sum_gamerounds", ax=axes[1], color="steelblue")

# Plot a histogram of game rounds for "gate_40" group on the third subplot
ab[(ab.version == "gate_40")].hist("sum_gamerounds", ax=axes[2], color="steelblue")

# Plot a boxplot comparing "gate_30" and "gate_40" game rounds on the fourth subplot
sns.boxplot(x=ab.version, y=ab.sum_gamerounds, ax=axes[3])

# Set overall title
plt.suptitle("After Removing Outliers", fontsize=20)
# Set subplot titles
axes[0].set_title("Distribution of Total Game Rounds", fontsize=15)
axes[1].set_title("Distribution of Gate 30 (A)", fontsize=15)
axes[2].set_title("Distribution of Gate 40 (B)", fontsize=15)
axes[3].set_title("Distribution of Both Groups", fontsize=15)

# Adjust spacing between subplots
plt.tight_layout(pad=4)


# Plot line chart of game rounds for "gate_30" group
# Reset index, set "index" column as new index, set legend and label to "Gate 30", and figure size to (20, 5)
ab[(ab.version == "gate_30")].reset_index().set_index("index").sum_gamerounds.plot(legend=True, label="Gate 30", figsize=(20, 5))

# Plot line chart of game rounds for "gate_40" group with alpha for transparency
ab[ab.version == "gate_40"].reset_index().set_index("index").sum_gamerounds.plot(legend=True, label="Gate 40", alpha=0.8)

# Set overall title
plt.suptitle("After Removing Outliers", fontsize=20)



fig, axes = plt.subplots(2, 1, figsize=(25, 10))

# Group by game rounds, count number of users per game round, and plot line chart on the first subplot
ab.groupby("sum_gamerounds").userid.count().plot(ax=axes[0])

# Group by game rounds, count number of users for the first 200 game rounds, and plot on the second subplot
ab.groupby("sum_gamerounds").userid.count()[:200].plot(ax=axes[1])

# Set the overall title
plt.suptitle("Number of Users at Each Game Round", fontsize=25)

# Set titles for subplots
axes[0].set_title("How many users at each game round?", fontsize=15)
axes[1].set_title("How many users in the first 200 game rounds?", fontsize=15)

# Adjust spacing between subplots
plt.tight_layout(pad=5)

# Display the top 20 game rounds with the highest number of users
ab.groupby("sum_gamerounds")['userid'].count().reset_index(name='user_cnt').sort_values('user_cnt', ascending=False)[:20]




# Get the number of users who reached level 30 and level 40
ab.groupby("sum_gamerounds").userid.count().loc[[30, 40]]


# Review summary statistics. The control and test groups appear similar,
# but are they statistically different? We'll investigate this using statistical methods.


# A/B Groups & Target Summary Stats
ab.groupby("version").sum_gamerounds.agg(["count", "median", "mean", "std", "max"])


# Create a 2x2 grid of subplots
fig, axs = plt.subplots(2, 2, figsize=(12, 10))

# Histogram
sns.histplot(data=ab, x='sum_gamerounds', hue='version', kde=False, ax=axs[0, 0])
axs[0, 0].set_title('Histogram')

# Density Plot
sns.kdeplot(data=ab, x='sum_gamerounds', hue='version', ax=axs[0, 1])
axs[0, 1].set_title('Density Plot')

# Box Plot
sns.boxplot(data=ab, x='version', y='sum_gamerounds', ax=axs[1, 0])
axs[1, 0].set_title('Box Plot')


# ECDF Plot
def ecdf(data):
    x = sorted(data)
    y = [i / len(x) for i in range(1, len(x) + 1)]
    return x, y

version_1_data = ab[ab['version'] == 'gate_30']['sum_gamerounds']
version_2_data = ab[ab['version'] == 'gate_40']['sum_gamerounds']

x_v1, y_v1 = ecdf(version_1_data)
x_v2, y_v2 = ecdf(version_2_data)

axs[1, 1].plot(x_v1, y_v1, label='gate_30')
axs[1, 1].plot(x_v2, y_v2, label='gate_40')
axs[1, 1].set_title('ECDF Plot')
axs[1, 1].legend()


plt.tight_layout()
plt.show()


pd.DataFrame({
    "RET1_COUNT": ab["retention_1"].value_counts(),  # Count of users retained on Day 1
    "RET1_RATIO": ab["retention_1"].value_counts() / len(ab),  # Proportion of users retained on Day 1
    "RET7_COUNT": ab["retention_7"].value_counts(),  # Count of users retained on Day 7
    "RET7_RATIO": ab["retention_7"].value_counts() / len(ab)   # Proportion of users retained on Day 7
})




ab.groupby(["version", "retention_1"]).sum_gamerounds.agg(["count", "median", "mean", "std", "max"])

ab.groupby(["version", "retention_7"]).sum_gamerounds.agg(["count", "median", "mean", "std", "max"])

stats1 = ab.groupby(["version", "retention_1"]).sum_gamerounds.agg(["count", "median", "mean", "std", "max"]).reset_index()


sns.set(style="whitegrid")
plt.figure(figsize=(16, 6))

# First subplot: Retention_1 Comparison by Version
plt.subplot(1, 2, 1)
ax1 = sns.barplot(data=stats1, x='version', y='count', hue='retention_1')
plt.title('Retention_1 Comparison by Version')
plt.xlabel('Version')
plt.ylabel('Count')

# Annotate exact values on the bars
for p in ax1.patches:
    ax1.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()),
                 ha='center', va='center', fontsize=10, color='black', xytext=(0, 5),
                 textcoords='offset points')

# Compute grouped stats by version and retention_7
stats7 = ab.groupby(["version", "retention_7"]).sum_gamerounds.agg(["count", "median", "mean", "std", "max"]).reset_index()

# Second subplot: Retention_7 Comparison by Version
plt.subplot(1, 2, 2)
ax2 = sns.barplot(data=stats7, x='version', y='count', hue='retention_7')
plt.title('Retention_7 Comparison by Version')
plt.xlabel('Version')
plt.ylabel('Count')

# Annotate exact values on the bars
for p in ax2.patches:
    ax2.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()),
                 ha='center', va='center', fontsize=10, color='black', xytext=(0, 5),
                 textcoords='offset points')

# Adjust spacing between subplots
plt.tight_layout()

# Show the plot
plt.show()

# Observations:
# We see similar results for users who played on Day 1 and Day 7 after installing the game.
# Roughly 12,000 users returned to play on both Day 1 and Day 7.

# Create a binary retention variable: 1 if retained on both Day 1 and Day 7, else 0
ab["Retention"] = np.where((ab.retention_1 == True) & (ab.retention_7 == True), 1, 0)

# Summary stats by version and new combined Retention flag
ab.groupby(["version", "Retention"])["sum_gamerounds"].agg(["count", "median", "mean", "std", "max"])

# Observation:
# Even when combining retention variables and comparing the groups, the summary statistics are still similar.

# Create a new column "NewRetention" by combining retention_1 and retention_7 as strings
ab["NewRetention"] = list(map(lambda x, y: str(x) + "-" + str(y), ab.retention_1, ab.retention_7))

# Group by "NewRetention" and "version" and aggregate game round statistics
ab.groupby(["NewRetention", "version"]).sum_gamerounds.agg(["count", "median", "mean", "std", "max"]).reset_index()

# This part of the code does the following:
# 1. Creates a new column "NewRetention" combining values of "retention_1" and "retention_7" with a hyphen.
# 2. Groups the data by "version" and "NewRetention".
# 3. Computes count, median, mean, std, and max of "sum_gamerounds" for each group.
# 4. Resets the index for readability and returns a tidy DataFrame.
# 
# Purpose: Analyze gameplay behavior based on combined Day 1 and Day 7 retention across A/B versions.

# === A/B Testing on Numeric Metrics ===

# Redefine A/B groups
# Replace "gate_30" with "A" and others (i.e., "gate_40") with "B"
ab["version"] = np.where(ab.version == "gate_30", "A", "B")

# Display the first few rows to verify the transformation
ab.head()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import shapiro, ttest_ind, mannwhitneyu, levene
import statsmodels.stats.api as sms

def AB_Test(dataframe, group, target, alpha=0.05, visualize=True):
    '''
    Perform an A/B test on the given numerical target variable between two groups ("A" and "B").

    Parameters:
    -----------
    dataframe : pd.DataFrame
        The data containing group and target columns.
    group : str
        Column name specifying A/B group labels ("A" and "B").
    target : str
        Column name for the numeric variable to test.
    alpha : float, default=0.05
        Significance level for hypothesis testing.
    visualize : bool, default=True
        Whether to visualize the distributions with histograms.

    Returns:
    --------
    result : pd.DataFrame
        Summary of statistical test results including p-value, test type, effect size, and power.
    '''

    # Split into A and B groups
    groupA = dataframe[dataframe[group] == "A"][target]
    groupB = dataframe[dataframe[group] == "B"][target]

    # Test for normality
    ntA = shapiro(groupA)[1] < alpha
    ntB = shapiro(groupB)[1] < alpha

    # Test for homogeneity of variances
    leveneTest = levene(groupA, groupB)[1] < alpha

    # Choose test type based on normality and variance assumptions
    if (not ntA) and (not ntB):  # Both normal
        if not leveneTest:
            ttest, p_value = ttest_ind(groupA, groupB, equal_var=True)
            test_type = "Parametric (Equal Variance)"
        else:
            ttest, p_value = ttest_ind(groupA, groupB, equal_var=False)
            test_type = "Parametric (Unequal Variance)"
    else:
        # Use non-parametric test if not normal
        ttest, p_value = mannwhitneyu(groupA, groupB, alternative='two-sided')
        test_type = "Non-Parametric"

    # Mean difference
    mean_diff = groupB.mean() - groupA.mean()

    # Effect size (Cohen's d)
    nobs_A = len(groupA)
    nobs_B = len(groupB)
    pooled_std = np.sqrt(((nobs_A - 1) * groupA.var() + (nobs_B - 1) * groupB.var()) / (nobs_A + nobs_B - 2))
    effect_size = mean_diff / pooled_std

    # Observed power
    observed_power = sms.tt_ind_solve_power(effect_size=effect_size, nobs1=nobs_A, alpha=alpha, alternative='two-sided')

    # Determine if result is statistically significant
    ab_hypothesis = p_value < alpha
    comment = "No significant difference between A and B." if not ab_hypothesis else "Significant difference between A and B!"

    # Format results as DataFrame
    result = pd.DataFrame({
        "Test Type": [test_type],
        "Target Variable": [target],
        "Equal Variance Assumed": ["Yes" if not leveneTest else "No"],
        "Hypothesis Result": ["Reject H0" if ab_hypothesis else "Fail to Reject H0"],
        "p-value": [p_value],
        "Mean Difference": [mean_diff],
        "Effect Size (Cohen's d)": [effect_size],
        "Observed Power": [observed_power],
        "Conclusion": [comment],
        "Test Statistic": [ttest]
    })

    # Print hypothesis statements and assumptions
    print("# A/B Test Hypotheses")
    print("H0: Group A == Group B")
    print("H1: Group A != Group B", "\n")
    print(f"Normality Assumption?\nA: {not ntA}, B: {not ntB}")
    print(f"Equal Variance Assumption: {not leveneTest}\n")

    # Optional: plot distribution
    if visualize:
        plt.figure(figsize=(6, 4))
        sns.histplot(data=dataframe, x=target, hue=group, kde=True)
        plt.axvline(groupA.mean(), color='blue', linestyle='dashed', linewidth=2, label=f'Mean A: {groupA.mean():.2f}')
        plt.axvline(groupB.mean(), color='orange', linestyle='dashed', linewidth=2, label=f'Mean B: {groupB.mean():.2f}')
        plt.title('Distribution of A/B Groups')
        plt.legend()
        plt.show()

    return result


result = AB_Test(dataframe=ab, group="version", target="sum_gamerounds")
result

df = ab.copy()
import sqlite3
conn = sqlite3.connect('abtest.db')

# Store the DataFrame into a SQLite database
df.to_sql('abtest', conn, if_exists='replace', index=False)

# Query data from the SQLite database
query = 'SELECT * FROM abtest'
df_from_db = pd.read_sql(query, conn)

# Display the queried DataFrame
df_from_db

query = '''


SELECT
    label,
    version,
    SUM(cnt) as cnts,
    SUM(is_retent) as retent_cnts
FROM (
    SELECT
        'retention_1' as label,
        version,
        userid,
        1 as cnt,
        retention_1 as is_retent
    FROM abtest

    UNION ALL

    SELECT
        'retention_7' as label,
        version,
        userid,
        1 as cnt,
        retention_7 as is_retent
    FROM abtest
) AS subquery
GROUP BY label, version;



'''

rent_res = pd.read_sql(query, conn)
rent_res

retention_1_A = df[df['version'] == 'A']['retention_1'].mean()
retention_1_B = df[df['version'] == 'B']['retention_1'].mean()

retention_7_A = df[df['version'] == 'A']['retention_7'].mean()
retention_7_B = df[df['version'] == 'B']['retention_7'].mean()

# Calculate the difference in retention rates (Delta)
delta_1 = retention_1_B - retention_1_A
delta_7 = retention_7_B - retention_7_A

# Compute sample sizes and proportions
n_A = df[df['version'] == 'A'].shape[0]  # Number of users in group A
n_B = df[df['version'] == 'B'].shape[0]  # Number of users in group B


retention_1_p1 = retention_1_A
retention_1_p2 = retention_1_B

retention_1_p = (retention_1_p1*n_A + retention_1_p2*n_B)/(n_A+n_B)

retention_7_p1 = retention_7_A
retention_7_p2 = retention_7_B

retention_7_p = (retention_7_p1*n_A + retention_7_p2*n_B)/(n_A+n_B)


se_1 = np.sqrt((retention_1_p*(1-retention_1_p) / n_A) + (retention_1_p*(1-retention_1_p) / n_B))
se_7 = np.sqrt((retention_7_p*(1-retention_7_p) / n_A) + (retention_7_p*(1-retention_7_p) / n_B))

# Calculate z-scores
z_1 = delta_1 / se_1
z_7 = delta_7 / se_7

# Calculate p-values (two-tailed test)
p_1 = stats.norm.sf(abs(z_1)) * 2  # Two-sided test for day-1 retention
p_7 = stats.norm.sf(abs(z_7)) * 2  # Two-sided test for day-7 retention

# Print results
print(f"Retention Day 1: Delta = {delta_1}, SE = {se_1}, z = {z_1}, p = {p_1}")
print(f"Retention Day 7: Delta = {delta_7}, SE = {se_7}, z = {z_7}, p = {p_7}")

# Assess statistical significance
if p_1 < 0.05:
    print("Day-1 retention difference is statistically significant.")
else:
    print("Day-1 retention difference is not statistically significant.")

if p_7 < 0.05:
    print("Day-7 retention difference is statistically significant.")
else:
    print("Day-7 retention difference is not statistically significant.")

alpha = 0.05
if p_1 < alpha:
    print("Retention 1-day difference is significant.")
else:
    print("Retention 1-day difference is not significant.")

if p_7 < alpha:
    print("Retention 7-day difference is significant.")
else:
    print("Retention 7-day difference is not significant.")

iterations = 500

bootstrap_df = pd.DataFrame()
p_values = []

df = ab.copy()
for x in tqdm(range(iterations)):
    iter_df = df.sample(frac = 1, replace=True).groupby(['version'], as_index=False).agg(
        retention_1 = ('retention_1', np.mean),
        retention_7 = ('retention_7', np.mean)
    )

    # control and variant data frames
    control_iter_df = iter_df[iter_df['version'] == 'A'].reset_index()
    variant_iter_df = iter_df[iter_df['version'] == 'B'].reset_index()

    bootstrap_df = pd.concat([bootstrap_df, iter_df])

# transpose data frame
bootstrap_df_melt = pd.melt(bootstrap_df, id_vars = 'version', value_vars = ['retention_1', 'retention_7'], var_name = 'ratio_metric')
bootstrap_df_melt.head()

# plot bootstrap distributions
plot = sns.FacetGrid(bootstrap_df_melt, col="ratio_metric", sharex = False, sharey = False, height = 4, hue = 'version')
plot.map_dataframe(sns.histplot, x="value", kde = True, stat = 'density', common_bins = True, fill = True)
plot.add_legend()

bootstrap_control = bootstrap_df[bootstrap_df['version'] == 'A'].reset_index()
bootstrap_variant = bootstrap_df[bootstrap_df['version'] == 'B'].reset_index()


# calculate KPIs differences
bootstrap_diffs = pd.DataFrame()

bootstrap_diffs['retention_1_diff'] = bootstrap_variant['retention_1'] - bootstrap_control['retention_1']
bootstrap_diffs['retention_7_diff'] = bootstrap_variant['retention_7'] - bootstrap_control['retention_7']


bootstrap_diffs_melt = pd.melt(bootstrap_diffs, value_vars = ['retention_1_diff', 'retention_7_diff'], var_name = 'ratio_metric')

# Define alpha for confidence intervals
alpha = 0.05
lower_ci = alpha / 2
upper_ci = 1 - (alpha / 2)

# Create a FacetGrid with seaborn
plot = sns.FacetGrid(bootstrap_diffs_melt, col="ratio_metric", sharex=False, height=6, aspect=1.2)
plot.map_dataframe(sns.histplot, x="value", kde=True, stat='probability', common_bins=True, fill=True)
plot.fig.suptitle(f'Bootstrapping differences with confidence intervals (alpha = {alpha})')

# Add confidence intervals lines
plot.map(lambda y, **kw: plt.axvline(y.quantile(lower_ci), color='r', linestyle='--'), 'value')
plot.map(lambda y, **kw: plt.axvline(y.quantile(upper_ci), color='r', linestyle='--'), 'value')

# Add confidence intervals annotations
plot.map(lambda y, **kw: plt.text(y.quantile(lower_ci), 0.04,
                                  f'Lower CI: {round(y.quantile(lower_ci), 4)}', color='k', ha='center'), 'value')
plot.map(lambda y, **kw: plt.text(y.quantile(upper_ci) * 1.125, 0.04,
                                  f'Upper CI: {round(y.quantile(upper_ci), 4)}', color='k', ha='center'), 'value')

# Add legend if needed
plot.add_legend()

# Show the plot
plt.show()


# Because 0 falls in confidence interval we can't say there is significant difference between variant groups.
# From above graphs we can see that there is a high probability that variant group will be worse. But what is the probability?


retention_1_prob = (bootstrap_diffs['retention_1_diff'] < 0).mean()
retention_7_prob = (bootstrap_diffs['retention_7_diff'] < 0).mean()

print('Probability that retention_1 in variant group will be worse than retention_1 in control group is', '{:.2%}.'.format(retention_1_prob))
print('Probability that retention_7 in variant group will be worse than retention_7 in control group is', '{:.2%}.'.format(retention_7_prob))



# Initialize lists to store bootstrap resampling results
boot_1d_diffs = []
boot_7d_diffs = []

# Set number of bootstrap iterations
n_bootstraps = 500  # You can increase to 5000 for more precision

# Perform bootstrap resampling
for i in tqdm(range(n_bootstraps)):
    boot_sample = df.sample(frac=1, replace=True)
    retention_1_A = boot_sample[boot_sample['version'] == 'A']['retention_1'].mean()
    retention_1_B = boot_sample[boot_sample['version'] == 'B']['retention_1'].mean()
    retention_7_A = boot_sample[boot_sample['version'] == 'A']['retention_7'].mean()
    retention_7_B = boot_sample[boot_sample['version'] == 'B']['retention_7'].mean()

    boot_1d_diffs.append(retention_1_B - retention_1_A)
    boot_7d_diffs.append(retention_7_B - retention_7_A)

# Convert results to Pandas Series
boot_1d_diffs = pd.Series(boot_1d_diffs)
boot_7d_diffs = pd.Series(boot_7d_diffs)

# Compute 95% confidence intervals
alpha = 0.05
ci_lower_1d = boot_1d_diffs.quantile(alpha / 2)
ci_upper_1d = boot_1d_diffs.quantile(1 - alpha / 2)
ci_lower_7d = boot_7d_diffs.quantile(alpha / 2)
ci_upper_7d = boot_7d_diffs.quantile(1 - alpha / 2)

# Print the confidence intervals
print(f"Retention Day 1: 95% CI = [{ci_lower_1d:.4f}, {ci_upper_1d:.4f}]")
print(f"Retention Day 7: 95% CI = [{ci_lower_7d:.4f}, {ci_upper_7d:.4f}]")

# Plot bootstrap distributions with confidence intervals
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

sns.histplot(boot_1d_diffs, kde=True, ax=ax1)
ax1.axvline(ci_lower_1d, color='r', linestyle='--')
ax1.axvline(ci_upper_1d, color='r', linestyle='--')
ax1.set_title('Bootstrapped Differences: 1-day Retention')
ax1.set_xlabel('Difference in Retention Rate')
ax1.set_ylabel('Frequency')

sns.histplot(boot_7d_diffs, kde=True, ax=ax2)
ax2.axvline(ci_lower_7d, color='r', linestyle='--')
ax2.axvline(ci_upper_7d, color='r', linestyle='--')
ax2.set_title('Bootstrapped Differences: 7-day Retention')
ax2.set_xlabel('Difference in Retention Rate')
ax2.set_ylabel('Frequency')

plt.tight_layout()
plt.show()

# Assess statistical significance based on whether CI excludes 0
if ci_lower_1d > 0 or ci_upper_1d < 0:
    print("Retention Day 1 difference is statistically significant.")
else:
    print("Retention Day 1 difference is NOT statistically significant.")

if ci_lower_7d > 0 or ci_upper_7d < 0:
    print("Retention Day 7 difference is statistically significant.")
else:
    print("Retention Day 7 difference is NOT statistically significant.")



# ================================================
# 🧾 Final Conclusion: Delta Method + Bootstrap
# ================================================

# Retention Day 1:
# - The difference in 1-day retention between the control (Gate 30) and test (Gate 40) groups is NOT statistically significant.
# - Interpretation: Moving the gate from level 30 to 40 does NOT significantly affect Day 1 retention.

# Retention Day 7:
# - The difference in 7-day retention IS statistically significant.
# - Interpretation: Moving the gate from level 30 to 40 DOES significantly impact Day 7 retention.

# -----------------------------------------------
# ✅ Final Verdict:
# Overall, under the given experimental setup,
# moving the first gate from level 30 to level 40 
# does not significantly affect user retention or total game rounds.
# This suggests that the change likely does NOT hurt player experience.
# -----------------------------------------------






