
# 📊 Cookie Cats A/B Testing Analysis

This project analyzes an A/B test conducted on **Cookie Cats**, a mobile puzzle game developed by Tactile Entertainment. The goal is to evaluate how shifting the **first level gate from level 30 to level 40** affects user behavior—specifically player **retention rates** and **game rounds played**.

---

## 🎮 Background

**Cookie Cats** is a match-3 style mobile game, similar to other puzzle games like *Candy Crush*. Players progress through levels, but occasionally face "gates"—checkpoints that require waiting or in-game purchases to continue.

To improve the game's retention and engagement, the developers ran an A/B test:

- **Group A (gate_30):** Players face the first gate at level 30 (control group).
- **Group B (gate_40):** Players face the first gate at level 40 (treatment group).

---

## 📌 Objectives

- Investigate whether changing the gate from level 30 to level 40 improves user **engagement** and **retention**.
- Use **data analysis**, **visualization**, and **statistical testing** to support decision-making.

---

## 📁 Dataset Description

The dataset includes **90,189 players** who installed the game during the A/B test period. Each row represents a player.

| Column           | Description                                                                  |
|------------------|------------------------------------------------------------------------------|
| `userid`         | Unique identifier for each player.                                           |
| `version`        | Group assignment: `gate_30` or `gate_40`.                                   |
| `sum_gamerounds` | Number of game rounds played in the first 7 days after installation.        |
| `retention_1`    | 1 if the player logged in the day after installation, else 0.               |
| `retention_7`    | 1 if the player logged in on the 7th day after installation, else 0.        |

---

## 🧠 Skills and Tools

- **Pandas** for data loading, cleaning, and transformation.
- **Seaborn & Matplotlib** for plotting distributions, bar plots, and retention trends.
- **Statistical Testing** to compare retention rates and game rounds between groups:
  - Two-sample t-tests
  - Proportion z-tests
- **Bootstrap Analysis** to estimate confidence intervals for retention differences.

---

## 📈 Key Analyses

1. **Exploratory Data Analysis (EDA):**
   - Distribution of game rounds (`sum_gamerounds`)
   - Retention rate differences (`retention_1`, `retention_7`)
   - Group-wise summary statistics

2. **Visualization:**
   - Histogram of rounds played
   - Bar plots for retention comparison

3. **Statistical Testing:**
   - Is the change from gate_30 to gate_40 statistically significant in improving day-1 or day-7 retention?
   - How does it affect the number of rounds played?

4. **Bootstrap (Optional Advanced):**
   - Estimate 95% confidence intervals for retention rate differences using resampling.

---

## 🧪 Expected Insights

- Does delaying the gate help keep players more engaged in the early stages?
- Are players more likely to return after 1 or 7 days?
- Is there a trade-off between increased gameplay and later frustration?

---

## 🧰 Getting Started

1. Install required packages:
   ```
   pip install pandas matplotlib seaborn scipy
   ```

2. Load and explore the data:
   ```python
   import pandas as pd
   df = pd.read_csv("cookie_cats.csv")
   ```

---

## 🧪 A Complete A/B Test Workflow

A well-structured A/B test typically includes the following steps:

1. **Analyze the Current Situation & Form Hypotheses**  
   Investigate the current business situation, identify the highest priority improvement opportunities, form hypotheses, and propose optimization suggestions.

2. **Define Metrics**  
   - **Primary Metrics** to evaluate the performance of different versions.
   - **Secondary Metrics** to assess potential side effects or additional impact.

3. **Design and Development**  
   Design a prototype for the optimized version and implement the changes.

4. **Determine Test Duration**  
   Establish how long the A/B test will run to ensure sufficient data is collected.

5. **Define Traffic Allocation Strategy**  
   Determine the percentage of users assigned to each group and other details of traffic splitting.

6. **Data Collection and Analysis**  
   Collect experimental data and evaluate both the validity and effectiveness of the test.

7. **Draw Conclusions**  
   Based on the results, decide whether to:
   - Roll out the new version to all users.
   - Adjust traffic split and continue testing.
   - Redesign and iterate based on insights, returning to step 1.
