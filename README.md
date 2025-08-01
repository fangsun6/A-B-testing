# 🎮 A/B Testing: Impact of Gate Level on Player Retention in Cookie Cats

This project investigates how changing the placement of the first in-game gate in the mobile puzzle game **Cookie Cats** affects player behavior—particularly retention and gameplay engagement—using robust statistical testing and visualization techniques.

---

## 📌 Problem Statement

Cookie Cats features “gates” that restrict player progression unless they wait or pay. Originally placed at **level 30**, Tactile Entertainment tested whether delaying the first gate to **level 40** would improve player **Day 1** and **Day 7 retention**, and **gameplay engagement**.

We aim to answer:
- ✅ Does moving the gate improve short- or long-term retention?
- 🎯 Does it alter how much players engage with the game?

---

## 📊 Dataset Overview

- **File**: `cookie_cats.csv`
- **Size**: ~90,000 rows × 5 columns
- **Columns**:
  - `userid`: Unique player ID
  - `version`: Group label (`gate_30` or `gate_40`)
  - `sum_gamerounds`: Number of rounds played in the first week
  - `retention_1`: Returned next day
  - `retention_7`: Returned a week later

---

## 📉 Key Metrics

- **Engagement**: `sum_gamerounds` — Total gameplay rounds
- **Retention**: Binary indicators `retention_1` and `retention_7`
- **Composite Metric**: Combined Day 1 and Day 7 retention

---

## 🧼 Data Cleaning & Preprocessing

### 🚧 Main Challenges:
| Challenge | Impact |
|----------|--------|
| **Extreme Outliers** | Some players logged 49k+ rounds—artificially inflated means and visual distortion |
| **Zero-round Players** | 4.4% of players didn’t play at all—skewed retention but irrelevant for engagement |
| **Heavy Skewness** | Most played <50 rounds, a few played 1000+ — mean is misleading |
| **Overplotting in Visuals** | Large volume made boxplots and histograms hard to interpret |

### 🧽 Cleaning Strategy:
| Step | Action | Rationale |
|------|--------|-----------|
| 1 | Removed top outlier (`max(sum_gamerounds)`) | Prevented single-player distortion |
| 2 | Used 1st–99th percentile range | Defined realistic player behavior |
| 3 | Treated zero-round users differently | Included for retention, excluded for engagement |
| 4 | Compared plots before/after cleaning | Verified improvement in clarity |

---

## 📊 Visualization & Interpretation

### 🔍 Visualization Goals:
- Understand group-level behavior
- Reveal skew and outliers
- Validate test assumptions
- Support hypothesis testing

### 🔧 Tools Used:
- **Histograms & Boxplots** — Engagement spread
- **Density/ECDF plots** — Retention distribution
- **Stacked Bar Charts** — Retention pattern breakdown

### 🎯 Observations:
- Most players quit early (low engagement)
- Minimal visual difference between groups post-cleaning
- Clear long-tail in engagement that needed trimming
- Day 7 retention differences only visible after aggregation

---

## 🧪 A/B Testing Design

| Group | Description |
|-------|-------------|
| **A** | Gate at level 30 (control) |
| **B** | Gate at level 40 (test) |

- **Randomized**: Players were randomly assigned
- **Binary & Numeric Metrics**: Analyzed both engagement and retention
- **Statistical Validity**: Checked assumptions before testing

---

## 🧠 Statistical Testing Strategy

### 1. Pre-Test Diagnostics
- **Shapiro-Wilk Test**: Normality
- **Levene's Test**: Variance equality

### 2. Statistical Test Selection:
| Conditions | Test Used |
|------------|-----------|
| Normal + Equal Variance | Student's t-test |
| Normal + Unequal Variance | Welch’s t-test |
| Non-normal | Mann–Whitney U test |

### 3. Bootstrapping (for robustness)
- 10,000 resamples to build sampling distribution
- Calculated confidence intervals and effect distributions

---

## 🧪 Binary Metric Testing (Retention)

Retention variables are binary (`True`/`False`) — we used:
- **Proportion comparison (Z-test)**:
  \
  z = (p1 - p2) / sqrt(p(1-p)(1/n1 + 1/n2))
  \
- **Bootstrapping**: To get confidence intervals on proportions

---

## 🔁 Bootstrapping Analysis: Why and How

### 🚧 Key Issues:
| Problem | Solution |
|---------|----------|
| **Non-normality** | Used resampling to build empirical distributions |
| **Outliers** | Trimmed extreme values, focused on medians |
| **Computational Cost** | Limited resamples, used `tqdm` for progress |
| **Small effect sizes** | Visualized effect size distribution to aid interpretation |

---

## 🧪 Experiment Results

### 📊 Engagement (Game Rounds)

- **Test Used**: Mann–Whitney U (non-parametric)
- **p-value**: 0.0509  
- ❌ No statistically significant difference at α = 0.05

### 📊 Retention (Day 1 and Day 7)

| Metric | Bootstrap 95% CI | Result |
|--------|------------------|--------|
| **Day 1** | [-0.002, 0.012] | ❌ Not significant |
| **Day 7** | [0.008, 0.022] | ✅ Statistically significant |

---

## 🎯 Final Conclusions

| Metric | Impact of Moving Gate to Level 40 |
|--------|----------------------------------|
| **Day 1 Retention** | No negative effect |
| **Day 7 Retention** | Moderate improvement (statistically significant) |
| **Engagement** | No significant change |

> 💡 **Key Takeaway**: Moving the gate delayed friction without harming short-term retention—and **possibly enhanced long-term loyalty**.

---

## 🤔 Reflection: Thinking Flow & Lessons Learned

### 🧠 Thinking Process
- Understand game design change → identify measurable KPIs
- Clean the data with statistical integrity
- Use EDA to form early hypotheses and guide test choices
- Choose proper tests based on assumptions, not habit
- Use bootstrapping to validate sensitive findings

### ⚠️ Project Challenges

| Challenge | Resolution |
|----------|------------|
| Skewed data | Used percentiles and log/box-cox inspection |
| Binary metrics | Z-tests and bootstrapped intervals |
| Interpretation | Combined visualization + statistical outputs |
| Visual clutter | Used filtered histograms, ECDFs, group splits |
| Marginal p-values | Validated with bootstrap CI, not just p-values |

---

## 📌 Summary

This A/B test shows how even small design changes (like level gating) can affect long-term user retention. A careful approach involving robust cleaning, tailored statistical testing, and bootstrapping helped ensure trustworthy conclusions.
