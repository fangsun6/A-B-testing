
# 📊 A/B Testing & Hash-Based Routing in Python

This repository documents two Python-based data science workflows:

1. **A/B Testing on Game Design** — measuring the impact of in-game gate placement on user retention.
2. **Hash-Based Routing System** — a backend-friendly method for assigning users to A/B/C groups in a deterministic and scalable way.

---

## 🧪 Part 1: A/B Testing Analysis – Cookie Cats Game

**File**: `AB_Testing_Converted.py`

### 🎮 Problem Context

In mobile gaming, developers often use "gates"—points where players must wait or pay—to control pacing. The business question was:

> _Will delaying the first gate from level 30 to level 40 improve player retention?_

This question inspired a well-designed A/B test.

---

### 🔍 Thought Process

- **Formulate Hypothesis**: If players encounter the gate later, they will be more engaged and more likely to return.
- **Design Metrics**: Focused on Day-1 and Day-7 retention, and total game rounds played.
- **Choose Tools**: Used Python libraries such as `pandas`, `matplotlib`, `scipy`, and `statsmodels`.

---

### 🧭 Step-by-Step Reasoning

1. **📥 Data Collection**
   - Used `gdown` to access a large public dataset of ~90,000 users.
   - Loaded data into a Pandas DataFrame and explored the schema.

2. **🧹 Data Cleaning**
   - Identified and removed users with 0 rounds played.
   - Detected and excluded extreme outliers using the 99th percentile.

3. **📊 EDA (Exploratory Data Analysis)**
   - Plotted histograms and ECDFs to compare engagement between gate_30 and gate_40.
   - Observed skewed data distributions, suggesting non-normality.

4. **🧪 Statistical Testing**
   - **Normality Test**: Shapiro-Wilk showed most variables were not normally distributed.
   - **Variance Check**: Used Levene’s test to verify assumptions.
   - **Choice of Test**: Employed Mann-Whitney U test over t-test due to skewed data.

5. **🔁 Bootstrap Inference**
   - Implemented bootstrap resampling to estimate confidence intervals of retention rates.
   - This non-parametric approach gave additional robustness.

6. **📈 Interpretation**
   - Visuals indicated a slight retention drop with gate_40.
   - Statistical tests confirmed that the drop was significant.

---

### 🧩 What Was Hard & How I Solved It

| Challenge | Thought Path & Solution |
|----------|--------------------------|
| Non-normal retention data | Switched to non-parametric Mann-Whitney U test |
| Binary outcome variable | Used proportions and bootstrap rather than relying solely on t-tests |
| Interpretation ambiguity | Combined visual (boxplots, histograms) with statistical rigor |
| User churn noise | Aggregated behavior over tens of thousands of users to reduce variance |
| Language support (Chinese fonts) | Embedded custom fonts into Matplotlib for proper visualization |

---

### ✅ Key Takeaways

- **Data volume helps but doesn’t guarantee clarity**—visuals and tests must work together.
- **Statistical testing is nuanced**—normality and variance assumptions matter.
- **Bootstrap is powerful**—great for inferring metrics from real-world data.

---

## 🔀 Part 2: Hash-Based Routing System

**File**: `hash_routing_cleaned.py`

### 🎯 Goal

Create a **stateless**, **fair**, and **consistent** method to assign users to experimental groups (A/B/C) without storing their IDs in a database.

---

### 🧠 Design Thinking

- Used MurmurHash3 (`mmh3`) because it’s fast, widely used in production, and has good randomness properties.
- Avoided modulo bias by applying `hash(user_id) % 100` with carefully split intervals:
  - A: 0–49
  - B: 50–74
  - C: 75–99

---

### 🔬 Verification Steps

1. **Simulated 10,000 random user IDs**
2. **Applied hash bucketing logic**
3. **Counted group assignment frequencies**
4. **Ran Chi-squared test** to validate statistical fairness
5. **Visualized distribution** using a pie chart

---

### 🧩 Challenges & Insights

| Challenge | Thought Path & Resolution |
|----------|---------------------------|
| How to ensure group balance? | Used Chi-square test against expected group proportions |
| What if the hash is biased? | Verified `mmh3` empirically over large random ID sets |
| Can we guarantee consistency? | Yes, same input → same hash → same group |
| Avoiding DB writes? | Stateless logic based on hash eliminates storage needs |

---

### 📉 Real-World Applications

- Feature flag rollouts
- Online experiments (A/B/n tests)
- Load balancing users across versions
- Personalization buckets

---

## 📁 Project Structure

```
.
├── AB_Testing_Converted.py     # Full A/B test analysis with stats and visualization
├── hash_routing_cleaned.py     # Stateless hash-based user bucketing with tests
└── README.md                   # Project overview, strategy, and reflections
```

---

## 📦 Installation

Install dependencies:

```bash
pip install pandas numpy matplotlib seaborn scipy statsmodels mmh3
```

---

## 📚 Final Reflections

This project integrates **applied statistics** and **practical software design**:

- A/B Testing: focuses on **hypothesis-driven experimentation**
- Hash Routing: focuses on **scalable and reproducible infrastructure**

Together, they demonstrate how data science is not just about models—but about solving real product and engineering problems with statistical rigor and implementation thinking.

---

## 📜 License

MIT License — use freely with attribution.
