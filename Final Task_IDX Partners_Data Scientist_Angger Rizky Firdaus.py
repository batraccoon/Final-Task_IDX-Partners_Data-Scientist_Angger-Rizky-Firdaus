#import the library used for this project
import opendatasets as od
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from feature_engine.outliers import Winsorizer
from scipy.stats import kendalltau, pearsonr, spearmanr
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import classification_report, recall_score, f1_score, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score
pd.set_option('display.max_columns', None)
from imblearn.over_sampling import SMOTENC
import warnings 
warnings.filterwarnings("ignore")
import pickle


# reading train data from CSV
csv = pd.read_csv(r'C:\Users\D5att\OneDrive\Documents\Rakamin\loan_data_2007_2014.csv',sep=',', on_bad_lines ="skip", index_col=False, dtype='unicode')

#Assign a new variable to display the dataset
df = csv.copy()

#dropping column
columns_drop = ['desc'                        ,
'next_pymnt_d',
'mths_since_last_record'      ,
'mths_since_last_delinq'         ,
'mths_since_last_major_derog' ,
'annual_inc_joint'            ,
'dti_joint'                   ,
'verification_status_joint'   ,
'open_acc_6m'                 ,
'open_il_6m'                  ,
'open_il_12m'                 ,
'open_il_24m'                 ,
'mths_since_rcnt_il'          ,
'total_bal_il'                ,
'il_util'                     ,
'open_rv_12m'                 ,
'open_rv_24m'                 ,
'max_bal_bc'                  ,
'all_util'                    ,
'inq_fi'                      ,
'total_cu_tl'                 ,
'inq_last_12m'                ]

df.drop(columns=columns_drop,axis=1,inplace=True)

#imputing specific column with 0 values
impute_columns = ['tot_coll_amt' , 'tot_cur_bal' , 'total_rev_hi_lim']

for kolom in impute_columns:
    df[kolom] = df[kolom].fillna(0)

 #drop the rest of null data
df.dropna(inplace=True)

#List of column based on the type of numerical data

int = ['Unnamed: 0','id','member_id','loan_amnt','funded_amnt','funded_amnt_inv','annual_inc','delinq_2yrs','inq_last_6mths','open_acc','pub_rec','revol_bal','total_acc','collections_12_mths_ex_med','policy_code','acc_now_delinq','tot_coll_amt','tot_cur_bal','total_rev_hi_lim']

flt = ['int_rate','installment','dti','revol_util','out_prncp','out_prncp_inv','total_pymnt','total_pymnt_inv','total_rec_prncp','total_rec_int','total_rec_late_fee','recoveries','collection_recovery_fee','last_pymnt_amnt']

#syntax for change the datatype
for integer in int:
    df[integer] = df[integer].astype('float64')
    df[integer] = df[integer].astype('int')

for float in flt:
    df[float] = df[float].astype('float')


high_cardinality_col = ['sub_grade', 'emp_title', 'issue_d', 'url', 'title', 'zip_code', 'addr_state', 'earliest_cr_line', 'last_pymnt_d', 'last_credit_pull_d']
df.drop(columns=high_cardinality_col,inplace=True,axis=1)


fully_paid = ['Fully Paid','Current','In Grace Period','Does not meet the credit policy. Status:Fully Paid']
default = ['Default','Charged Off','Late (31-120 days)','Late (16-30 days)','Does not meet the credit policy. Status:Charged Off']

#The target value has already been combined and encoded.
df['loan_status'] = df['loan_status'].replace(fully_paid, 0)
df['loan_status'] = df['loan_status'].replace(default, 1)

#separation of features and targets
X = df.drop(['loan_status'], axis = 1)
y = df['loan_status']


#separation between train dataset and test dataset.

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.40, random_state = 30)
print('Train Size: ', X_train.shape)
print('Test Size: ', X_test.shape)


#separation between test dataset and test dataset.

X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size = 0.50, random_state = 30)
print('Val Size: ', X_val.shape)
print('Test Size: ', X_test.shape)

skew = ['id',
 'member_id',
 'loan_amnt',
 'funded_amnt',
 'funded_amnt_inv',
 'installment',
 'annual_inc',
 'inq_last_6mths',
 'open_acc',
 'revol_bal',
 'total_acc',
 'out_prncp',
 'out_prncp_inv',
 'total_pymnt',
 'total_pymnt_inv',
 'total_rec_prncp',
 'total_rec_int',
 'last_pymnt_amnt',
 'tot_cur_bal',
 'total_rev_hi_lim']

normal = [ 'Unnamed: 0', 'int_rate', 'dti', 'revol_util','delinq_2yrs', 'pub_rec', 'total_rec_late_fee', 'recoveries', 'collection_recovery_fee', 'collections_12_mths_ex_med', 'acc_now_delinq', 'tot_coll_amt']

#syntax for handling outliers by capping using winsorizer for gaussian capping method


winsoriserskew = Winsorizer(capping_method='iqr',
                            tail='both',
                            fold=1.5,
                            variables= skew,
                            missing_values='ignore')

X_train_capped = winsoriserskew.fit_transform(X_train)
X_test_capped = winsoriserskew.transform(X_test)

#syntax for handling outliers by capping using winsorizer for gaussian capping method

winsorisernorm = Winsorizer(capping_method='gaussian',
                            tail='both',
                            fold=3,
                            variables= normal,
                            missing_values='ignore')

X_train_capped = winsorisernorm.fit_transform(X_train)
X_test_capped = winsorisernorm.transform(X_test)

categorical_features = [6,9,10,11,13,14,15,24,37]

smote_nc = SMOTENC(categorical_features=categorical_features, random_state=42)

# Apply SMOTENC to the training data
X_train_resampled, y_train_resampled = smote_nc.fit_resample(X_train, y_train)
# Display the resampled data
print("Resampled feature matrix:\n", X_train_resampled.shape)
print("Resampled target vector:\n", y_train_resampled.shape)

#Assign ColumnTransformer to perform scaling and encoding on columns that have been determined in feature engineering
#Scaling using standardscaler and encoding using onehotencoder
prep = ColumnTransformer([
    ('scaler', StandardScaler(), [ 'loan_amnt',
 'funded_amnt',
 'funded_amnt_inv',
 'int_rate',
 'installment',
 'annual_inc',
 'dti',
 'delinq_2yrs',
 'inq_last_6mths',
 'open_acc',
 'pub_rec',
 'revol_bal',
 'revol_util',
 'total_acc',
 'out_prncp',
 'out_prncp_inv',
 'total_pymnt',
 'total_pymnt_inv',
 'total_rec_prncp',
 'total_rec_int',
 'total_rec_late_fee',
 'recoveries',
 'collection_recovery_fee',
 'last_pymnt_amnt',
 'collections_12_mths_ex_med',
 'acc_now_delinq',
 'tot_coll_amt',
 'tot_cur_bal',
 'total_rev_hi_lim']),
    ('encoding', OneHotEncoder(handle_unknown='ignore'),['term',
 'grade',
 'emp_length',
 'home_ownership',
 'verification_status',
 'purpose',
 'initial_list_status'])])       

pipe_RF = Pipeline([
('transformer', prep),
('classifier', RandomForestClassifier())
])

#the model is fitted to the train data
pipe_RF.fit(X_train_resampled, y_train_resampled)

#the model learns X test data
y_RF_pred_train = pipe_RF.predict(X_train_resampled)
y_RF_pred_test = pipe_RF.predict(X_test_capped)

#model saving

with open('model.pkl', 'wb') as file_1:
  pickle.dump(pipe_RF, file_1)