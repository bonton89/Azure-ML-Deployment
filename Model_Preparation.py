# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.6.0
#   kernelspec:
#     display_name: Python 3.6 - AzureML
#     language: python
#     name: python3-azureml
# ---

# +
import pandas as pd



def clean_missing_data(df):
    df.drop(['Loan_ID','Gender','Property_Area'],axis=1,inplace=True)
    numerical_col = df.select_dtypes(include = ['number']).columns
    categorical_col = df.select_dtypes(include = ['object','category','bool']).columns
    for i in numerical_col:
        df[i] = df[i].fillna(df[i].mean())
    for j in categorical_col:
        df[j] = df[j].fillna(df[j].mode()[0])
        
    return df


def preprocessing(df):
    
    from sklearn.preprocessing import MinMaxScaler
    
    X = df.drop('Loan_Status',axis=1)
    Y = df[['Loan_Status']]
    
    num_col = X.select_dtypes(include=['number']).columns
    cat_col = X.select_dtypes(include = ['object','bool','category']).columns
    scaler = MinMaxScaler()
    scaler_fitted = scaler.fit(X[num_col])
    df[num_col] = scaler_fitted.transform(X[num_col])
    
    X = pd.get_dummies(X,columns = cat_col,drop_first = True)
    
    Y = Y.replace({True:1,False:0})
    
    return X, Y




def train(X,Y):
    
    from sklearn.model_selection import train_test_split
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3, random_state = 1234, stratify=Y)
    
    from sklearn.ensemble import RandomForestClassifier
    rfc = RandomForestClassifier(random_state=123)
    model = rfc.fit(X_train, Y_train)
    Y_predict = rfc.predict(X_test)
    
    Y_prob = rfc.predict_proba(X_test)[:, 1]
    
    from sklearn.metrics import confusion_matrix
    
    Confusion_Matrix    = confusion_matrix(Y_test, Y_predict)
    Score = rfc.score(X_test, Y_test)
    
   
    return model,Confusion_Matrix,Score
# -


