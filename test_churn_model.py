import unittest
from pathlib import Path
import pandas as pd
from churn_model import evaluate,prepare_input
class ChurnTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.data=pd.read_csv(Path(__file__).with_name('Telco-Customer-Churn.csv'))
 def test_holdout_and_learned_transformations(self):
  report,model,train,test=evaluate(self.data)
  self.assertFalse(set(train)&set(test))
  self.assertEqual(len(train)+len(test),len(self.data))
  self.assertEqual(sum(map(sum,report['logistic_regression']['confusion_matrix'])),len(test))
  X,_=prepare_input(self.data)
  learned=model.named_steps['prepare'].named_transformers_['numeric'].named_steps['impute'].statistics_
  expected=X.loc[train,['tenure','MonthlyCharges','TotalCharges']].median().values
  for a,b in zip(learned,expected): self.assertAlmostEqual(a,b)
 def test_duplicate_customers_and_missing_labels_rejected(self):
  with self.assertRaisesRegex(ValueError,'Duplicate'): prepare_input(pd.concat([self.data,self.data.iloc[:1]]))
  bad=self.data.copy();bad.loc[0,'Churn']=None
  with self.assertRaisesRegex(ValueError,'Churn must'): prepare_input(bad)
if __name__=='__main__':unittest.main()
