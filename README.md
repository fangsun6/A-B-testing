
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

- **Source**: [Google Drive - Public CSV Download](https://drive.google.com/uc?id=1IkN-fylT9ZYxZJhzbgpAmJ751b6lrPy9)
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


## 🧪 Before Removing Extreme Values

### 🔍 Raw Player Activity (Line Plot)

![Before Removing The Extreme Value](./before_extreme_line.png)

- Each point represents a player's weekly game rounds.
- A huge spike in `gate_30` shows one user with nearly 50,000 rounds.

### 📊 Histogram & Boxplot View

![Group Distributions Before Filtering](./before_extreme_hists.png)

- **Left**: Histogram for `gate_30`
- **Center**: Histogram for `gate_40`
- **Right**: Boxplot comparison of both groups

These visuals justified filtering outliers before applying statistical tests or comparing means.

---


## 🛠 Project Workflow

1. **Load Data & Clean**
2. **Visualize Engagement**
3. **Compute Retention**
4. **Run Tests (Shapiro, Levene, T-tests, Bootstrap)**
5. **Draw Business Insights**

---

## 📈 Visualizations

- Histograms, boxplots, ECDFs
- Retention comparisons
- Bootstrap distributions

---

## 💡 Key Insights

- Moving the gate to level 40 **decreased retention** slightly.
- Engagement remained similar across groups.
- Retention metrics were **statistically different** via bootstrap & tests.

---

## 📬 Contact

- 👤 Your Name
- 📧 your.email@example.com
- 🔗 [GitHub](https://github.com/yourname)
