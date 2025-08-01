
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

## 🧼 Data Cleaning & Preprocessing

### ⚠️ Key Challenges:

1. **Extreme Outliers in Gameplay Rounds**
2. **Zero-Engagement Users**
3. **Highly Skewed Distributions**

### 🔍 Cleaning Strategy:

| Step | Action Taken | Rationale |
|------|--------------|-----------|
| 1.   | Removed users with `sum_gamerounds == max()` | Outliers distorted averages. |
| 2.   | Examined 1st–99th percentiles | Guided decision on thresholding. |
| 3.   | Visualized before and after | Plotted histograms and boxplots. |
| 4.   | Treated zero-round users selectively | Included for retention, excluded from gameplay plots. |

---

![Uploading Unknown-3.png…]()


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
