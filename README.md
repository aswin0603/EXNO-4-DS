# EXNO:4-DS
# AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

# ALGORITHM:
STEP 1:Read the given Data.

STEP 2:Clean the Data Set using Data Cleaning Process.

STEP 3:Apply Feature Scaling for the feature in the data set.

STEP 4:Apply Feature Selection for the feature in the data set.

STEP 5:Save the data to the file.

# FEATURE SCALING:
1. Standard Scaler: It is also called Z-score normalization. It calculates the z-score of each value and replaces the value with the calculated Z-score. The features are then rescaled with x̄ =0 and σ=1
2. MinMaxScaler: It is also referred to as Normalization. The features are scaled between 0 and 1. Here, the mean value remains same as in Standardization, that is,0.
3. Maximum absolute scaling: Maximum absolute scaling scales the data to its maximum value; that is,it divides every observation by the maximum value of the variable.The result of the preceding transformation is a distribution in which the values vary approximately within the range of -1 to 1.
4. RobustScaler: RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value).

# FEATURE SELECTION:
Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.
The feature selection techniques used are:
1.Filter Method
2.Wrapper Method
3.Embedded Method

# CODING AND OUTPUT:

```python
import pandas as pd
from scipy import stats
import numpy as np
df = pd.read_csv("/content/bmi.csv")
df.head()
```
![image](https://github.com/user-attachments/assets/e006da88-715f-4716-9268-bdafea75324d)

```python
df_null_sum = df.isnull().sum()
df_null_sum
```
![image](https://github.com/user-attachments/assets/98495198-2d1a-4ae4-934a-d25cfab41b2e)

```python
df.dropna()
```
![image](https://github.com/user-attachments/assets/f6adc765-ce7d-4107-aa4c-63614c21e179)

```python
max_vals = np.max(np.abs(df[['Height', 'Weight']]), axis=0)
max_vals
```
![image](https://github.com/user-attachments/assets/3778b134-4f95-4736-8050-333f237ba5d9)

### Standard Scaling
```python
from sklearn.preprocessing import StandardScaler
df1=pd.read_csv("/content/bmi.csv")
df1.head()
```
![image](https://github.com/user-attachments/assets/92e86de0-8ae8-456a-9ecc-a80aa81a922b)

```python
sc=StandardScaler()
df1[['Height','Weight']]=sc.fit_transform(df1[['Height','Weight']])
df1.head(10)
```
![image](https://github.com/user-attachments/assets/0daee022-90f1-4f78-824a-b4acc3d1ee00)

### Min-Max Scaling
```python
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
df[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']])
df.head(10)
```
![image](https://github.com/user-attachments/assets/5fa8c726-812b-4680-ba3f-dd3492389f59)

### Maximum Absolute Scaling

```python
from sklearn.preprocessing import MaxAbsScaler
scaler = MaxAbsScaler()
df3=pd.read_csv("/content/bmi.csv")
df3.head()
```
![image](https://github.com/user-attachments/assets/f2d985a4-b4d5-4555-8985-8c2a4c813e88)

```python
df3[['Height','Weight']]=scaler.fit_transform(df3[['Height','Weight']])
df3
```
![image](https://github.com/user-attachments/assets/4466284d-f286-42bd-8ed4-e7cceb07b201)

### Robust Scaling
```python
from sklearn.preprocessing import RobustScaler
scaler = RobustScaler()
df4=pd.read_csv("/content/bmi.csv")
df4.head()
```
![image](https://github.com/user-attachments/assets/3c402a10-c0c1-4b5b-a14d-6c3a3810388b)

```python
df4[['Height','Weight']]=scaler.fit_transform(df4[['Height','Weight']])
df4.head()
```
![image](https://github.com/user-attachments/assets/48ac1c69-8ec6-4da9-a752-3aa860ba26cb)

### Feature Selection

```python
df=pd.read_csv("/content/income(1) (1).csv")
df
```
![image](https://github.com/user-attachments/assets/facc2480-1299-4e13-bd52-3bb8509ab2bc)

```python
df.info()
```
![image](https://github.com/user-attachments/assets/9c68f794-010a-4944-bd07-35552e2e4ebb)

```python
df_null_sum=df.isnull().sum()
df_null_sum
```
![image](https://github.com/user-attachments/assets/b1e2e692-d3db-4061-9681-edeccd23f90b)

```python
categorical_columns = ['JobType', 'EdType', 'maritalstatus', 'occupation', 'relationship', 'race', 'gender', 'nativecountry']
df[categorical_columns] = df[categorical_columns].astype('category')
df[categorical_columns] = df[categorical_columns].apply(lambda x: x.cat.codes)
df[categorical_columns]
```
![image](https://github.com/user-attachments/assets/a6e4a6ca-b3e1-43f9-a034-1b6eb0e5ecee)

```python
X = df.drop(columns=['SalStat'])
y = df['SalStat']
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
```
![image](https://github.com/user-attachments/assets/c375a434-eebe-4b1f-99e6-0d5ef8557514)

```python
y_pred = rf.predict(X_test)
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy using selected features: {accuracy}")
```
![image](https://github.com/user-attachments/assets/f660794b-7242-4ef4-86da-c907e016ba70)

### Filter Method

```python
import pandas as pd
from sklearn.feature_selection import SelectKBest, chi2, f_classif
categorical_columns = ['JobType', 'EdType', 'maritalstatus', 'occupation', 'relationship', 'race', 'gender', 'nativecountry']
df[categorical_columns] = df[categorical_columns].astype('category')
df[categorical_columns]
```
![image](https://github.com/user-attachments/assets/16617cb3-c75b-4625-9010-c38a466fa173)

```python
df[categorical_columns] = df[categorical_columns].apply(lambda x: x.cat.codes)
df[categorical_columns]
```
![image](https://github.com/user-attachments/assets/ff351789-a987-4cff-94f4-b50641c076b2)

```python
X = df.drop(columns=['SalStat'])
y = df['SalStat']
k_chi2 = 6
selector_chi2 = SelectKBest(score_func=chi2, k=k_chi2)
X_chi2 = selector_chi2.fit_transform(X, y)
selected_features_chi2 = X.columns[selector_chi2.get_support()]
print("Selected features using chi-square test:")
print(selected_features_chi2)
```
![image](https://github.com/user-attachments/assets/2b9487ca-c42b-46da-b4cc-34a1c6117f0d)

```python
selected_features = ['age', 'maritalstatus', 'relationship', 'capitalgain', 'capitalloss',
'hoursperweek']
X = df[selected_features]
y = df['SalStat']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
```
![image](https://github.com/user-attachments/assets/2cb59340-da26-4d72-b222-cfb12854a302)

```python
y_pred = rf.predict(X_test)
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy using selected features: {accuracy}")
```
![image](https://github.com/user-attachments/assets/acbc89be-b5d4-4b3b-9239-42d67d0607b9)

### Fisher Square

```python
pip install skfeature-chappers
```
![image](https://github.com/user-attachments/assets/594b0846-878a-49ed-a468-3c8443aacae2)

```python
import numpy as np
import pandas as pd
from skfeature.function.similarity_based import fisher_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
categorical_columns = ['JobType', 'EdType', 'maritalstatus', 'occupation', 'relationship', 'race', 'gender', 'nativecountry']
df[categorical_columns] = df[categorical_columns].astype('category')
df[categorical_columns]
```
![image](https://github.com/user-attachments/assets/311ddff2-0f35-485a-9695-afe32bf59eda)

```python
df[categorical_columns] = df[categorical_columns].apply(lambda x: x.cat.codes)
df[categorical_columns]
```
![image](https://github.com/user-attachments/assets/fe2469fb-7bd7-440e-a73e-b9f42b4832aa)




































# RESULT:
Thus we have read the given data and performed Feature Scaling and Feature Selection process and saved the data to the file.


### Name: Aswin B
### Register Number: 212224110007
