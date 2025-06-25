
# 📊 A/B Testing & Hash-Based Routing in Python

This repository contains two key Python projects:

1. **A/B Testing on Game Design** — measuring the impact of in-game gate placement on user retention.
2. **Hash-Based Routing System** — a backend-friendly method for assigning users to A/B/C groups in a deterministic and fair way.

---

## 🧪 Part 1: A/B Testing Analysis - Cookie Cats Game

**File**: `AB_Testing_Converted.py`

### 🎮 Context & Objective

In mobile games, balancing monetization and user experience is critical. *Cookie Cats* uses “gates” at certain levels to encourage in-app purchases or impose waiting time. The hypothesis here is: _Does moving the first gate from level 30 to level 40 affect player retention?_

We aim to evaluate this with a full-cycle A/B testing workflow.

---

### 🧠 Analytical Thoughts

- **Retention is critical**: Early player engagement (Day-1 and Day-7 retention) often predicts long-term success.
- **Gate timing might frustrate or retain**: A gate too early could repel users; too late may reduce monetization opportunities.

---

### 🛠️ Workflow Overview

1. **Data Collection**:
   - Pulled data from Google Drive using `gdown`.
   - Dataset includes: `user_id`, `version` (A or B), `sum_gamerounds`, `retention_1`, `retention_7`.

2. **Data Cleaning**:
   - Checked for zero-play users.
   - Removed extreme outliers (max game rounds).

3. **Exploratory Data Analysis**:
   - Used histograms, boxplots, ECDF plots, and bar charts to visualize group differences.
   - Customized fonts for Chinese visualization (TaipeiSans).

4. **Statistical Testing**:
   - Shapiro-Wilk test → Most data was non-normal.
   - Used Levene’s test to check variance equality.
   - Employed t-tests or Mann-Whitney U where appropriate.

5. **Bootstrap**:
   - Used resampling to estimate confidence intervals for retention metrics.

---

### 🧩 Key Difficulties & Solutions

| Problem | Approach |
|--------|----------|
| **Non-normal data** | Used non-parametric tests (Mann-Whitney U) |
| **Skewed game rounds** | Removed extreme outliers (>99th percentile) |
| **Retention is binary** | Focused on proportions and bootstrap analysis |
| **User behavior is noisy** | Aggregated large sample (90K+ users) to ensure robustness |
| **Conflicting visual vs. statistical results** | Relied on statistical inference for rigor |

---

### ✅ Outcome

- The change **negatively affected Day-1 retention** (statistically significant).
- Recommendation: **Keep the gate at level 30** to maintain early user engagement.
- Illustrated the importance of data-driven game design decisions.

---

## 🔀 Part 2: Hash-Based Routing

**File**: `hash_routing_cleaned.py`

### 🎯 Purpose

In A/B testing, you need a consistent and fair way to assign users into groups (A/B/C). Storing these assignments in a database adds overhead. Instead, we use **hash functions** to assign users **deterministically**.

---

### 💡 Concept

Using a hash function (`mmh3`) on user IDs ensures:
- **Consistency**: Same user always gets the same group.
- **Fairness**: Groups can be balanced by percentage allocation (50% A, 25% B, 25% C).
- **Statelessness**: No need to persist the assignment in a DB.

---

### 📈 Analysis Process

1. **Generate 10,000 random user IDs**.
2. **Hash each ID** using `mmh3` and reduce `mod 100`.
3. **Assign to groups** based on pre-defined ranges:
   - A: 0–49
   - B: 50–74
   - C: 75–99

4. **Run Chi-Square test** to compare actual vs expected ratios.

---

### 🧩 Challenges & Mitigation

| Challenge | Strategy |
|----------|----------|
| **Biased hash function** | Used `mmh3`, a high-quality, fast hash function |
| **Imbalanced sample** | Tested over large random samples to check distribution |
| **Reproducibility** | `mmh3` provides consistent hashing across runs |
| **User tracking without storage** | Solved by deterministic hashing instead of DB mapping |

---

### 📊 Output

- Pie chart visualization confirms near-equal proportions.
- Chi-square test confirms **no statistically significant difference** from expected group sizes.

---

## 📁 File Structure

```
.
├── AB_Testing_Converted.py     # Complete A/B testing notebook with plots and analysis
├── hash_routing_cleaned.py     # Script for hash-based deterministic user bucketing
└── README.md                   # Project explanation, reasoning, and instructions
```

---

## 📦 Requirements

Install dependencies:

```bash
pip install pandas numpy matplotlib seaborn scipy statsmodels mmh3
```

---

## 💭 Final Thoughts

Both projects reflect key components of data science:
- The **A/B test** focuses on experiment design, statistical rigor, and business decision-making.
- The **hash router** focuses on scalable engineering, fairness, and automation.

This repo bridges both analysis and deployment concerns in a real-world data-driven workflow.

---

## 📜 License

MIT License. Provided for educational purposes.
