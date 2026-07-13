# 🌾 Crop Recommendation System

A machine learning project that recommends the most suitable crop to grow based on soil and climate conditions. Built as part of my Machine Learning coursework.

This README isn't just a summary — it's basically a walkthrough of how I actually approached the problem, what I tried, and what I learned along the way.

---

## The Idea

Farmers usually decide what to grow based on experience, tradition, or guesswork. But soil and climate conditions can be measured — nitrogen, phosphorus, potassium levels, temperature, humidity, pH, rainfall — so why not let data decide (or at least assist) which crop would actually thrive?

That's the whole premise here: feed a model these seven numbers, and it tells you which crop is the best fit.

---

## The Dataset

`data.csv` — 2,200 rows, 8 columns, 22 different crops (100 samples each, nicely balanced).

| Column | Meaning |
|---|---|
| N | Nitrogen content in soil |
| P | Phosphorous content in soil |
| K | Potassium content in soil |
| temperature | Temperature (°C) |
| humidity | Relative humidity (%) |
| ph | Soil pH |
| rainfall | Rainfall (mm) |
| label | Crop name (target) |

Crops range from rice, banana, and mango to lentils, cotton, and coffee — a nice mix of grains, fruits, and cash crops.

---

## My Process (a.k.a. the actual journey)

### 1. First, just look at the data
Before touching any model, I loaded the CSV and did the basics: `head()`, `shape`, `describe()`, `info()`. Nothing fancy — just getting a feel for the numbers. 2,200 rows, 8 columns, all numeric except the crop label.

### 2. Check for problems
Ran `isnull().sum()` — zero missing values across the board. That was a relief; no messy imputation needed. Also checked `.apply(lambda x: len(x.unique()))` to see how varied each column was, and confirmed the label column had 22 unique, evenly distributed crop classes (`value_counts()` — exactly 100 samples per crop).

### 3. Explore before assuming
This is the part I didn't want to skip. I plotted:
- A **correlation heatmap** to see how N, P, K, temperature, humidity, pH, and rainfall relate to each other.
- **Distribution plots** for every feature (N, P, K, temperature, humidity, pH, rainfall) to see if anything was skewed or had weird outliers.
- A **count plot** of the crop labels, just to double check the class balance visually instead of trusting the numbers blindly.

Nothing screamed "problem" — the features looked reasonably well-behaved, and the classes were balanced, which meant I didn't need to worry about oversampling/undersampling tricks.

### 4. Prep the data for modeling
- Split the data into features (`X` = the 7 soil/climate columns) and target (`y` = crop label).
- The label was text (like `"rice"`, `"banana"`), so I ran it through `LabelEncoder` to turn crop names into numbers the models could actually work with.
- Did an 80/20 `train_test_split` (with `random_state=42` for reproducibility).

### 5. Try more than one model
Instead of committing to a single algorithm right away, I tested three different approaches to see which one actually understood the patterns best:

- **Decision Tree** — simple, interpretable, good baseline.
- **Logistic Regression** — a classic linear approach, mostly to see how it'd handle a multi-class problem like this.
- **Random Forest** — an ensemble method, expected to do a bit better by averaging out the noise a single tree might overfit to.

Each one was fit on the training set and scored on the held-out test set.

### 6. Compare, don't assume
I didn't just eyeball the accuracy numbers — I stored them in lists and plotted a bar chart to compare all three models side by side. That way the difference (however small) is actually visible, not just a number I skimmed past.

---

## Results

| Model | Accuracy |
|---|---|
| Decision Tree | **98.86%** |
| Logistic Regression | 94.55% |
| Random Forest | **99.32%** |

Random Forest came out on top — which honestly makes sense. It's an ensemble of trees, so it tends to generalize a bit better than a single Decision Tree and definitely outperformed the linear assumption baked into Logistic Regression on a problem like this, where the relationship between soil/climate features and crop type isn't purely linear.

Decision Tree was surprisingly close behind, which says the data has fairly clean, separable patterns per crop — not a lot of noisy overlap between classes.

Logistic Regression, while still solid at ~94.5%, struggled relatively more, likely because it assumes linear decision boundaries, and real-world agricultural data doesn't always play by those rules. (It also threw a convergence warning during training — a sign it could use more iterations or scaled features if I wanted to push it further.)

---

## What I'd Improve Next

If I revisit this project, here's what's on my list:
- **Scale the features** (StandardScaler) before Logistic Regression — might close the accuracy gap.
- **Try cross-validation** instead of a single train/test split, to be more confident the results aren't just a lucky split.
- **Hyperparameter tuning** for Random Forest (`n_estimators`, `max_depth`, etc.) — squeezing out extra performance.
- **Feature importance** analysis — which of N, P, K, temperature, humidity, pH, or rainfall actually matters most per crop? Would make the model more explainable to an actual farmer.
- **Try other models** — XGBoost or KNN could be interesting comparisons.

---

## Files in This Project

```
├── crop-recommendation.ipynb   # Full notebook: EDA, preprocessing, model training & comparison
├── data.csv                    # Dataset (2200 rows, 22 crop classes)
└── README.md                   # This file
```

---

## Tools Used

- **Python** (pandas, numpy)
- **Seaborn / Matplotlib** for visualization
- **Scikit-learn** for preprocessing (`LabelEncoder`, `train_test_split`) and modeling (`DecisionTreeClassifier`, `LogisticRegression`, `RandomForestClassifier`)

---

## Takeaway

This project was less about chasing the highest accuracy number and more about the process — looking at the data honestly before modeling it, testing more than one algorithm instead of assuming one is "the" answer, and actually comparing results instead of just trusting whichever model finished training first. Random Forest won this round, but the bigger lesson was in the loop itself: explore → clean → split → try → compare → reflect.
