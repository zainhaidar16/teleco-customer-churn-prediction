"""Reproducible Telco baseline; all learned preprocessing is fit on training rows only."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, confusion_matrix, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERIC = ['tenure','MonthlyCharges','TotalCharges']
SEED = 42

def prepare_input(frame):
    data = frame.copy()
    required = ['Churn','customerID',*NUMERIC]
    missing = [c for c in required if c not in data]
    if missing: raise ValueError('Missing required columns: ' + ', '.join(missing))
    if data.customerID.isna().any() or data.customerID.astype(str).str.strip().eq('').any():
        raise ValueError('Each row needs a customerID.')
    if data.customerID.astype(str).str.strip().duplicated().any():
        raise ValueError('Duplicate customer IDs: resolve repeated snapshots before splitting.')
    labels = data.Churn.astype(str).str.strip().map({'No':0,'Yes':1})
    if labels.isna().any(): raise ValueError('Churn must contain only Yes or No; missing labels are not imputed.')
    if len(data)<40 or labels.value_counts().min()<10 or labels.nunique()!=2:
        raise ValueError('Provide at least 40 rows and 10 examples of each churn class.')
    features = data.drop(columns=['Churn','customerID'])
    for column in NUMERIC: features[column] = pd.to_numeric(features[column], errors='coerce')
    features[NUMERIC] = features[NUMERIC].replace([np.inf,-np.inf],np.nan)
    categorical = [c for c in features if c not in NUMERIC]
    for column in categorical: features[column] = features[column].fillna('Unknown').astype(str)
    return features, labels.astype(int)

def metrics(y, probability):
    prediction = probability >= 0.5
    return {'accuracy':float(accuracy_score(y,prediction)), 'precision':float(precision_score(y,prediction,zero_division=0)), 'recall':float(recall_score(y,prediction,zero_division=0)), 'average_precision':float(average_precision_score(y,probability)), 'roc_auc':float(roc_auc_score(y,probability)), 'confusion_matrix':confusion_matrix(y,prediction,labels=[0,1]).tolist()}

def evaluate(frame):
    X,y = prepare_input(frame)
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,stratify=y,random_state=SEED)
    category_columns = [c for c in X if c not in NUMERIC]
    preprocessing = ColumnTransformer([
        ('numeric',Pipeline([('impute',SimpleImputer(strategy='median',keep_empty_features=True)),('scale',StandardScaler())]),NUMERIC),
        ('category',OneHotEncoder(handle_unknown='ignore'),category_columns)
    ])
    model = Pipeline([('prepare',preprocessing),('model',LogisticRegression(max_iter=2000,random_state=SEED))])
    model.fit(X_train,y_train)
    baseline = DummyClassifier(strategy='prior').fit(X_train,y_train)
    report = {'seed':SEED,'threshold':0.5,'split':'stratified 80% training / 20% test','train_rows':len(X_train),'test_rows':len(X_test),'test_churn_prevalence':float(y_test.mean()),'logistic_regression':metrics(y_test,model.predict_proba(X_test)[:,1]),'prior_baseline':metrics(y_test,baseline.predict_proba(X_test)[:,1])}
    return report,model,X_train.index.tolist(),X_test.index.tolist()

if __name__ == '__main__':
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('csv',type=Path)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    report,*_=evaluate(pd.read_csv(args.csv))
    result=json.dumps(report,indent=2)
    if args.output: args.output.write_text(result+'\n',encoding='utf-8')
    print(result)
