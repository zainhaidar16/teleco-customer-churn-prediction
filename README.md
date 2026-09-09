# Telco churn: an evaluated baseline

This portfolio experiment uses the bundled Telco CSV. Its acquisition/version and reuse terms need to be documented before redistributing beyond this existing repository. It is not a live retention system.

## Reproduce
Use Python 3.12 in a virtual environment.

    pip install -r requirements.txt
    python -m unittest -v test_churn_model.py
    python churn_model.py Telco-Customer-Churn.csv --output evaluation.json
    streamlit run streamlit_app.py

## Evaluation design
Unique customer IDs are required; repeated customer snapshots are rejected to prevent cross-split leakage. Target labels are validated and never imputed. A deterministic, stratified 80/20 train/test split uses seed 42. Numeric imputation, scaling, and category encoding are learned only from training rows. Unknown test categories are handled explicitly.

Logistic regression is compared with a class-prior baseline. The report includes test precision, recall, average precision, ROC-AUC, accuracy, prevalence, and a confusion matrix (rows actual; columns predicted; labels No, Yes). Accuracy alone can reward predicting the majority class.

The fixed threshold is 0.50. A real retention decision needs intervention cost, expected saved margin, treatment effectiveness, and separate validation data for threshold selection. Avoid repeated tuning against the test set. A random split does not demonstrate generalization across time or companies, and predicting churn does not prove an intervention prevents it.

## Audit change
The old neural-network app labelled training accuracy as Final Accuracy and prepared data before its validation split. This revision establishes a simpler, auditable baseline with a true holdout. TensorFlow/Keras are no longer used. No performance or business impact is claimed without a saved evaluation result.
