# Predictive Factors Linked to Student Dropout or Completion

A PySpark decision tree model that predicts whether a student drops out or completes their studies, based on enrolment, background, academic and economic factors.

Completed as part of a project at Brunel University.

## Overview
- **Goal:** identify and predict student dropout vs. completion.
- **Target variable:** `newTarget` (binary: 0 / 1) [state what 0 and 1 mean, e.g. 1 = completed].
- **Model:** Decision Tree Classifier (Spark MLlib).
- **Environment:** Google Colab, Python 3, PySpark 3.4.0.

## Data
- **File:** `student_cleaned_data.csv` (cleaned version of the original dataset)
- **Source:** [dataset name and link]
- **Size:** 29 columns [add number of rows]
- **Feature groups:**
  - **Demographic:** marital status, nationality, gender, age at enrolment, displaced, educational special needs
  - **Family background:** mother's and father's qualification and occupation
  - **Application and course:** application mode, application order, course, day/evening attendance, previous qualification
  - **Financial:** debtor, tuition fees up to date, scholarship holder
  - **Academic performance:** first and second semester credits, enrolled units, evaluations, grades, units without evaluation
  - **Economic context:** unemployment rate, inflation rate, GDP

## Method
1. Load and inspect the cleaned CSV with pandas.
2. Rename columns (dots replaced with underscores) and convert to a Spark DataFrame.
3. Index features with `StringIndexer` and combine them with `VectorAssembler`.
4. Train a `DecisionTreeClassifier` in a Spark `Pipeline`.
5. Split the data 70% training and 30% testing.
6. Evaluate on the test set.

## Results
| Metric | Score |
|--------|-------|
| Accuracy | 0.862 |
| Weighted precision | 0.863 |
| Weighted recall | 0.862 |
| Training time | 36.26 s |

### Interpretation
- The decision tree correctly classified about **86% of students** in the held-out 30% test set, so it separates dropouts from completers well.
- Precision (0.863) and recall (0.862) are almost the same as accuracy. The model is balanced, with no sign that it favours one outcome over the other.
- Training took about **36 seconds** on Google Colab, which is quick for a dataset of this size.
- The scores are **weighted averages** across both classes. If one class is much larger than the other, they can hide weaker performance on the smaller class (usually dropouts).

### Caveats
- The data was split once at random with no fixed seed, so scores may vary slightly between runs.
- Features such as second-semester grades and evaluations are recorded after enrolment. They are strong predictors but may not be available for early-warning use.
- The results show association, not causation.

## How to run
1. Open the notebook in Google Colab.
2. Run the first cells and upload `student_cleaned_data.csv` when prompted.
3. Run the remaining cells in order.

Dependencies: `pandas`, `numpy`, `seaborn`, `matplotlib`, `missingno`, `pyspark`

## Limitations
- Results show association, not causation.
- The split is random with no fixed seed, so results vary slightly between runs.
- Only one model was tested.

## Acknowledgements
Dataset credit: [Open Sources -Surveys, Student Records and Institutional Data].
