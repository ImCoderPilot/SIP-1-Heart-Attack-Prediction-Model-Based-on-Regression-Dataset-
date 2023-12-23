import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
# Load the dataset
data = pd.read_csv('/heart.csv')

# Examine the dataset
print(data.head())
   age  sex  cp  trtbps  chol  fbs  restecg  thalachh  exng  oldpeak  slp  \
0   63    1   3     145   233    1        0       150     0      2.3    0   
1   37    1   2     130   250    0        1       187     0      3.5    0   
2   41    0   1     130   204    0        0       172     0      1.4    2   
3   56    1   1     120   236    0        1       178     0      0.8    2   
4   57    0   0     120   354    0        1       163     1      0.6    2   

   caa  thall  output  
0    0      1       1  
1    0      2       1  
2    0      2       1  
3    0      2       1  
4    0      2       1
import pandas as pd
# Load the dataset
data = pd.read_csv('/heart.csv')
# Display the column names
print(data.columns)

Index(['age', 'sex', 'cp', 'trtbps', 'chol', 'fbs', 'restecg', 'thalachh',
       'exng', 'oldpeak', 'slp', 'caa', 'thall', 'output'],
      dtype='object'
# Split into features (X) and target variable (y)
X = data[['age', 'cp', 'trtbps', 'chol' , 'fbs' , 'restecg' , 'thalachh' , 'exng' , 'sex' , 'oldpeak' , 'slp' , 'caa' , 'thall' , 'output']]  
y = data['output']  # Replace 'target' with the actual target variable name


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Linear Regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error (MSE): ", mse)
print("R-squared (R2) score: ", r2)
Mean Squared Error (MSE):  1.6004347150856172e-30
R-squared (R2) score:  1.0
# Example prediction on heart Attack
heart_Attack = pd.DataFrame({'age': [63],  'cp' : [3] , 'trtbps': [145] , 'chol' : [233] , 'fbs' : [1], 'restecg' : [0], 'thalachh' : [150], 'exng'  : [0], 'sex' : [1] , 'oldpeak' : [2.3], 'slp' : [0] , 'caa': [0], 'thall' : [1], 'output' : [1]})  # Replace values with actual data
prediction = model.predict(heart_Attack)
if prediction==0 :
  print("Prediction  is  less chance of heart attack ", prediction)
else :
   print("prediction is more chance of heart attack", prediction)
prediction is more chance of heart attack [1.]
