## Prediction code for "Supervised machine learning algorithms for predicting student dropout and academic success: a comparative study" by Villar & Andrade

[Paper source](https://link.springer.com/article/10.1007/s44163-023-00079-z)

[Original Github Repository](https://github.com/alicevillar/SML-Comparative-Study)

[Dataset Source](https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success)

#### Contains the dataset:
- **Predict Students' Dropout and Academic Success:** Predicting whether a university student will drop out, remain enrolled, or graduate based on demographic data, socio-economic factors, and academic performance.

*(This dataset originates from a European higher education institution. It contains highly sensitive demographic and socio-economic quasi-identifiers such as the parents' background, whether the student is a debtor, gender, and age at enrollment. A linkage attack on this dataset presents a severe privacy risk, exposing an individual's financial distress and academic failure.)*

---

## Adaptations and changes from the original code

Original Jupyter Notebook: `SML-Code.ipynb` from the [SML-Comparative-Study GitHub Repository](https://github.com/alicevillar/SML-Comparative-Study/blob/main/SML-Code.ipynb)

#### The Role of `optuna` in Our Methodology:

To match the original authors' methodology, we included the `optuna` hyperparameter optimization framework in our pipeline. We use it to find the best configuration for the  `RandomForestClassifier`. Specifically, it identifies the `max_depth`, `min_samples_leaf`, and `criterion` parameters using 10-fold cross-validation optimized for the `f1_weighted` score.

#### Code Modifications for our Study:
WWhile we kept the hyperparameter optimization to ensure a valid baseline, we made a few changes to fit the strict requirements of our privacy evaluation:

1. **Determinism and Baseline Freezing:** 
The original Optuna setup uses a non-deterministic sampler by default, which causes the baseline accuracy to fluctuate slightly between runs. Since we need a completely stable environment to measure predictive utility after anonymization, we set a fixed seed (`optuna.samplers.TPESampler(seed=42)`). This allows us to find the optimal hyperparameters on the raw data once, and then lock in the model architecture for a fair 1-to-1 comparison during the privacy tests.
2. **Isolation of Baseline Utility:** The original code uses SMOTE and ADASYN resampling methods to artificially balance the minority classes before training. We skipped this resampling step entirely. To ensure synthetic data generation wouldn't skew our privacy metrics, we trained the Random Forest model strictly on the natural, real-world data distribution.