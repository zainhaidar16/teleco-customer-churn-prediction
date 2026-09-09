from pathlib import Path
import pandas as pd
import streamlit as st
from churn_model import evaluate

st.set_page_config(page_title='Telco churn: an evaluated baseline',layout='wide')
st.title('Telco churn: an evaluated baseline')
st.write('Compare logistic regression with a class-prior baseline using an independent 20% test split. Imputation, scaling, and encoding are fitted only on training data.')
st.caption('Portfolio experiment. This does not establish retention impact or production readiness.')
upload=st.file_uploader('Optional Telco-format CSV',type=['csv'])
st.write('Use the bundled Telco sample, or upload a compatible file with unique customerID values and a Yes/No Churn target.')
if st.button('Run evaluation',type='primary'):
    try:
        frame=pd.read_csv(upload if upload is not None else Path(__file__).with_name('Telco-Customer-Churn.csv'))
        with st.spinner('Fitting on training data and evaluating the test split…'):
            report,*_=evaluate(frame)
        st.subheader('Held-out test results')
        st.caption(str(report['train_rows'])+' training rows · '+str(report['test_rows'])+' test rows · fixed threshold 0.50')
        comparison=pd.DataFrame({name:{k:v for k,v in report[name].items() if k!='confusion_matrix'} for name in ['logistic_regression','prior_baseline']}).T
        st.dataframe(comparison.style.format('{:.3f}'))
        st.subheader('Confusion matrix')
        st.dataframe(pd.DataFrame(report['logistic_regression']['confusion_matrix'],index=['Actual: stays','Actual: churns'],columns=['Predicted: stays','Predicted: churns']))
        st.info('The 0.50 threshold is a fixed evaluation choice, not an optimized retention policy. Select any operational threshold on separate validation data using intervention cost, expected saved margin, and treatment effectiveness. Do not tune using these test results.')
        st.json(report)
    except (ValueError,TypeError,KeyError) as error:
        st.error(str(error))
