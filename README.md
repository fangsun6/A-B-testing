
# 🎮 A/B Testing: Impact of Gate Level on Player Retention in Cookie Cats

This project conducts an A/B test analysis using Python to investigate how the placement of a gate in the mobile game **Cookie Cats** affects player retention and engagement.

---

## 📌 Problem Statement

**Cookie Cats**, a puzzle game developed by Tactile Entertainment, includes progression "gates" at certain levels. Players must either wait or pay to advance beyond these gates. The original gate was placed at **level 30**, but the company wanted to test whether moving it to **level 40** would improve retention.

We use data from 90,000+ players to explore:

- 💡 Does moving the gate to level 40 improve player retention on Day 1 or Day 7?
- 📉 How does the gate change affect overall gameplay behavior (e.g., number of rounds played)?

---

## 🧾 Dataset

- **Source**: cookie_cats.csv
- **Size**: 90,189 observations, 5 columns
- **Format**: CSV

---

## 📄 Sample Data

| userid | version | sum_gamerounds | retention_1 | retention_7 |
|--------|---------|----------------|-------------|-------------|
| 116    | gate_30 | 3              | False       | False       |
| 337    | gate_30 | 38             | True        | False       |
| 377    | gate_40 | 165            | True        | False       |
| 483    | gate_40 | 1              | False       | False       |
| 488    | gate_40 | 179            | True        | True        |

---

## 📊 Group Summary Statistics

| version  | count | median | mean   | std      | max   |
|----------|-------|--------|--------|----------|--------|
| gate_30  | 44700 | 17     | 52.46  | 256.72   | 49854 |
| gate_40  | 45489 | 16     | 51.30  | 103.29   | 2640  |

---


## 🧼 Data Cleaning, Preprocessing & Visualizing Engagement

Understanding player behavior required a careful approach to both data preparation and exploratory visualization.

---

### ⚠️ Key Data Challenges:

1. **Extreme Outliers in Gameplay Rounds**  
   Some players logged over 49,000 rounds in one week, severely distorting averages and plots.

2. **Zero-Engagement Users**  
   ~4.4% of users never played a round. Including them impacted retention metrics but not engagement analysis.

3. **Skewed Distributions**  
   Data was highly right-skewed, with most players playing <50 rounds but a few playing thousands.

4. **Overplotting and Clutter**  
   With 90k+ users, some line or scatter plots became unreadable.

---

### 🔍 Cleaning & Visualization Strategy:

| Step | Action Taken | Rationale |
|------|--------------|-----------|
| 1.   | Removed users with `sum_gamerounds == max()` | Prevent single-user spikes from skewing all plots |
| 2.   | Reviewed percentiles | Identified a realistic cap for “normal” users |
| 3.   | Visualized before & after | Compared group histograms and boxplots pre/post-cleaning |
| 4.   | Filtered zero-round users selectively | Included for retention, excluded for engagement comparisons |

---

### 📈 Visualization Techniques Used:

- **Histograms & Boxplots**  
  Revealed the shape and spread of player engagement.

- **Density Plots & ECDFs**  
  Used for comparing group-wide behavioral distributions.

- **Stacked Bar Plots**  
  Highlighted breakdowns in 1-day and 7-day retention.

---

### 🧠 What We Learned:

- Most users engaged very lightly — outliers masked key trends.
- Median values provided more useful comparisons than means.
- Cleaning the data before visualization was critical for accurate, interpretable results.

---|--------------|-----------|
| 1.   | Removed users with `sum_gamerounds == max()` | Outliers distorted averages. |
| 2.   | Examined 1st–99th percentiles | Guided decision on thresholding. |
| 3.   | Visualized before and after | Plotted histograms and boxplots. |
| 4.   | Treated zero-round users selectively | Included for retention, excluded from gameplay plots. |




## 📊 Visualizing Engagement

After cleaning the data, visualizing user engagement (number of rounds played) was essential to uncover hidden patterns and guide further analysis.

### 🎯 Goals

- Compare player activity across A/B test groups
- Understand overall engagement behavior
- Detect skewness and abnormal user behavior
- Support statistical testing decisions with visuals

---

### ⚠️ Challenges Encountered

1. **Severe Skewness in Distribution**
   - Most players played fewer than 100 rounds, but a small number played hundreds or even thousands — producing long right tails.
   - This made **mean values misleading**, especially before removing outliers.

2. **Overplotting in Boxplots**
   - Extreme outliers crowded the visual space, flattening the boxplot body and obscuring typical engagement levels.
   - Boxplot visualization was clearer only after outliers were removed.

3. **Group Comparison**
   - Initial histograms overlaid both groups, but small differences (like a 1-round median difference) were hard to see.
   - Separate subplots helped isolate and compare group behavior more effectively.

---

### ✅ Solutions Applied

- Used **histograms**, **boxplots**, and **ECDF plots** to capture both global and fine-grained patterns
- Applied **log-transformation and zooming** for sanity checks on tail behavior (not included in final charts but useful for exploration)
- Separated plots by group (`gate_30` vs `gate_40`) for clarity
- Annotated key values and thresholds to aid interpretability

---

### 📌 Key Takeaway

Visualizing engagement revealed:
- High early-game churn (many players had very few rounds)
- A long-tail distribution caused by a few highly active users
- Minimal visual difference in gameplay between the two groups after cleaning — aligning with statistical findings

These insights were crucial for understanding player behavior beyond what summary statistics could tell.


## 🔁 Computing Retention Metrics

Player retention is a critical metric in mobile gaming, reflecting how well the game keeps users engaged after installation. In this project, we focused on **Day 1** and **Day 7 retention**.

---

### 📌 Retention Definitions:

- **Day 1 Retention (`retention_1`)**  
  Whether the player opened the app again on the **day after install**.

- **Day 7 Retention (`retention_7`)**  
  Whether the player opened the app again **seven days after install**.

Each variable is a boolean column (`True` or `False`) in the dataset.

---

### 🧠 Why It Matters

- Retention is closely tied to user experience and monetization.
- A/B testing retention helps quantify the impact of game design decisions (e.g., moving the gate).

---

### 📈 How We Computed and Analyzed Retention:

1. **Group Aggregation**  
   We grouped players by `version` (gate_30 vs gate_40) and calculated the **mean** of `retention_1` and `retention_7`.  
   Since the retention columns are boolean, the mean directly gives the retention rate.

2. **Visual Comparisons**  
   We used bar plots and stacked bar plots to compare retention between groups.

3. **Combined Retention Flag**  
   We also created a combined variable:  
   ```python
   ab["Retention"] = np.where((ab.retention_1 == True) & (ab.retention_7 == True), 1, 0)
   ```
   This allowed us to analyze users who returned both on Day 1 and Day 7.

4. **Retention Breakdown Matrix**  
   By combining `retention_1` and `retention_7` into a single label (e.g., `"True-False"`), we created a 2x2 segmentation of player behavior.

---

### 📊 Insights from Retention Analysis

- The test group (`gate_40`) showed a **slight decrease in Day 1 retention** compared to the control group.
- Day 7 retention was low overall, consistent with industry trends.
- Combining retention labels revealed that most retained users returned only on Day 1, not on both days.

Retention metrics gave a clear, quantifiable view of how changing the game gate impacted long-term player engagement.

## 🧪 A/B Testing Design and User Group Structure

The core of this analysis revolves around an A/B test that compared two different versions of the game, differing only in the placement of the first gate (obstacle):

- **Group A (Control):** First gate appears at level 30  
- **Group B (Test):** First gate appears at level 40

The goal is to determine whether moving the gate improves or harms user retention and engagement.

---

### 🧬 Sample User Data Structure

Below is a preview of the dataset after transformation:

| userid | version | sum_gamerounds | retention_1 | retention_7 | Retention | NewRetention |
|--------|---------|----------------|-------------|-------------|-----------|---------------|
| 116    | A       | 3              | False       | False       | 0         | False-False   |
| 337    | A       | 38             | True        | False       | 0         | True-False    |
| 377    | B       | 165            | True        | False       | 0         | True-False    |
| 483    | B       | 1              | False       | False       | 0         | False-False   |
| 488    | B       | 179            | True        | True        | 1         | True-True     |

- **`version`**: Converted to 'A' or 'B' for easier labeling.
- **`Retention`**: Binary flag where 1 = player returned on both Day 1 and Day 7.
- **`NewRetention`**: Combines both retention flags into a string to capture full user behavior (`True-True`, `True-False`, etc.)

---

### 🧪 A/B Testing Workflow

1. **Define the Test Variable**  
   - The placement of the first progression gate: level 30 vs level 40.

2. **Random Assignment (Pre-defined)**  
   - Players were pre-assigned to either `gate_30` or `gate_40`.

3. **Convert Group Labels**  
   - Mapped `gate_30` → A, and `gate_40` → B for clarity.

4. **Analyze Key Metrics**  
   - Engagement: `sum_gamerounds`
   - Retention: `retention_1`, `retention_7`, and their combinations

5. **Test for Statistical Significance**  
   - Applied normality and variance tests
   - Compared group metrics using T-tests, Welch tests, or Mann-Whitney U tests as appropriate

---

### 🎯 Goal

To determine whether the change introduced in group B (delayed gate) causes a statistically significant difference in player behavior compared to group A.

The carefully structured A/B test, combined with binary and composite retention labels, enables robust analysis of both short-term and long-term engagement outcomes.

---

### 📐 A/B Test Statistical Testing Procedure

To rigorously determine whether the differences observed between Group A and Group B were statistically significant, we followed a structured testing process:

#### 1. Split & Define Groups
- Group A: Users with the gate at level 30
- Group B: Users with the gate at level 40
- Variables analyzed: `sum_gamerounds`, `retention_1`, `retention_7`

#### 2. Apply Shapiro-Wilk Test
- Purpose: Check if the engagement or retention variable is normally distributed within each group.
- Result determines whether parametric tests (e.g., t-test) are appropriate.

#### 3. Apply Levene’s Test (if normality holds)
- Purpose: Test whether the variances between groups are equal.
- Required for using the standard t-test.

#### 4. Choose the Correct Test Based on Results:

| Condition                                | Test Used               |
|------------------------------------------|--------------------------|
| Normality & equal variances              | **Student’s t-test**     |
| Normality & unequal variances            | **Welch’s t-test**       |
| Non-normal distribution                  | **Mann–Whitney U test**  |

#### 5. Bootstrap Analysis (Optional/Complementary)
- Used to estimate the sampling distribution of the difference in means or retention rates.
- Helped confirm test results when distribution assumptions were questionable.

---

### 🧠 Why This Matters

- Choosing the wrong test can lead to **false positives** or **false negatives**.
- Non-parametric tests like Mann–Whitney U are more robust but less powerful.
- Bootstrap inference provided additional confidence and intuition about effect size.

This systematic procedure ensured that the conclusions drawn from the A/B test were statistically sound and reproducible.

---

## 📊 A/B Testing for 0-1 (Binary) Metrics

For binary outcome variables—like player retention—the analysis involves comparing two proportions (i.e., conversion or retention rates). These variables take on only two values (0 or 1), making them suitable for proportion-based hypothesis testing.

### 🧮 Examples of Binary Metrics

- **Day 1 Retention Rate** = (Number of users who returned on Day 1) / (Total users)
- **Cold Start Success Rate** = (Number of campaigns that reached 20 conversions in 3 days) / (Total new campaigns)
- **Conversion Rate** = (Users who took action) / (Total exposed users)

Each of these metrics can be treated as a binary variable per user (success or failure), and collectively analyzed using proportion tests.

---

### 🧪 Proportion Testing Method

In A/B testing, we compare the proportions of success between two independent groups (e.g., Group A and Group B). These follow a **binomial distribution**, but under large sample sizes, the sampling distribution of proportions approximates a **normal distribution**, allowing us to use a **Z-test**.

#### Z-test Formula for Two Proportions:

\[
z = \frac{p_1 - p_2}{\sqrt{p(1-p)(\frac{1}{n_1} + \frac{1}{n_2})}}
\]

Where:
- **p₁, p₂**: Proportions observed in Group A and Group B (e.g., retention rates)
- **n₁, n₂**: Sample sizes of Group A and B
- **p**: Combined proportion, computed as:
  \[
  p = \frac{n_1 p_1 + n_2 p_2}{n_1 + n_2}
  \]
- **(1 − p)**: The complementary proportion of the combined sample

---

### 🔍 When to Use This

- When your metric is binary per user (e.g., retained or not)
- When you're interested in testing **rates** across groups
- When you have a large enough sample size for normal approximation

---

### 🧠 Why It Matters

This method is commonly used in product analytics to test the impact of UI/UX changes, feature introductions, or other interventions on key metrics like:

- Click-through rate (CTR)
- Activation or signup rate
- Retention or churn rate

Accurate testing of binary metrics ensures that even small changes in proportions are statistically validated and not due to random chance.


---

## 🔁 Bootstrapping Analysis

To validate our A/B test results—especially for retention and engagement metrics—we used **bootstrapping**, a powerful resampling method that helps estimate uncertainty without relying on strong parametric assumptions.

---

### 💡 What is Bootstrapping?

Bootstrapping involves:

1. Randomly resampling the dataset with replacement
2. Calculating a statistic (e.g., mean, difference in means, proportion) on each resample
3. Repeating this process thousands of times to create a distribution of that statistic

From this bootstrapped distribution, we can derive:
- Confidence intervals
- Probability of difference (e.g., how often B outperforms A)

---

### ⚠️ Challenges Faced

1. **Non-normal Distributions**  
   Metrics like `sum_gamerounds` and retention rates were not normally distributed, violating assumptions of classic parametric tests.

2. **Skewed Data & Outliers**  
   Bootstrapped results can be highly sensitive to extreme values. We dealt with this by:
   - Removing outliers during preprocessing
   - Comparing trimmed means and medians as more robust alternatives

3. **Small Effect Sizes**  
   When group differences are small, classic tests might not detect them clearly. Bootstrapping provided more nuanced insights by visualizing the distribution of differences.

4. **Performance Cost**  
   Bootstrapping is computationally intensive. To balance accuracy and speed:
   - We used 10,000 resamples
   - Applied progress tracking with tools like `tqdm`

---

### 🧠 Key Insights from Bootstrapping

- Showed that even when means are close, distributions may differ in shape or confidence bounds
- Reinforced statistical significance where classic tests showed borderline p-values
- Helped interpret uncertainty in real-world impact, not just hypothesis test results

---

### ✅ When to Use Bootstrapping

- Your data is non-normal or heavily skewed
- You’re working with medians or quantiles
- You want visual intuition for confidence in group differences

Bootstrapping gave this A/B test added depth by quantifying the certainty of our conclusions with fewer assumptions.

---

## 🧾 Experimental Results & Conclusion

We conducted non-parametric and bootstrap-based analyses to assess the effect of moving the first gate from level 30 to level 40 in the Cookie Cats game. The outcomes were evaluated on two key metrics: engagement (game rounds) and retention.

---

### 🎮 Numeric Metric: Game Rounds

- **Test Type:** Two-sided non-parametric test
- **p-value:** 0.0509

At the conventional significance level (α = 0.05), we **failed to reject the null hypothesis**. This suggests that we **do not have sufficient evidence** to claim a significant difference in total game rounds played between Group A (gate_30) and Group B (gate_40).

---

### 📈 Ratio Metric: Retention

We used the **Bootstrap method** to evaluate the stability and significance of differences in retention:

#### 🧪 Bootstrapping Steps:

1. Conducted 500 resamples with replacement from the original dataset
2. Calculated average `retention_1` and `retention_7` in each resample for both control and test groups
3. Computed the difference in retention between groups for each iteration
4. Constructed 95% confidence intervals from the bootstrapped distribution

#### 📊 Bootstrap Results:

- **Retention Day 1:** 95% CI = [-0.002, 0.012]  
  ➤ Not statistically significant (CI includes 0)

- **Retention Day 7:** 95% CI = [0.008, 0.022]  
  ➤ Statistically significant (CI excludes 0)

---

### ✅ Final Conclusion

- **Day 1 Retention:** No significant difference between control and test groups → moving the gate did not affect short-term retention.
- **Day 7 Retention:** Significant improvement in the test group → moving the gate positively affected long-term retention.

Overall, moving the first gate from level 30 to level 40 **did not negatively impact** player experience, and may even **improve long-term engagement** without disrupting short-term play behavior.
