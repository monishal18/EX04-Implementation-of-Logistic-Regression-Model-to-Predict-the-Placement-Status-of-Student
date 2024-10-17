# EX 4 Implementation of Logistic Regression Model to Predict the Placement Status of Student
## DATE:

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Data Collection and Preprocessing
2. Algorithm for Logistic Regression Prediction
3. Explanation of Key Components
4. Evaluate the model

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: monisha.L
RegisterNumber: 2305001019 
*/
import pandas as pd
 data=pd.read_csv("/content/ex45Placement_Data.csv")
 data.head()
 data1=data.copy()
 data1.head()
 data1=data1.drop(['sl_no','salary'],axis=1)
 data1
 from sklearn.preprocessing import LabelEncoder
 le=LabelEncoder()
 data1["gender"]=le.fit_transform(data1["gender"])
 data1["ssc_b"]=le.fit_transform(data1["ssc_b"])
 data1["hsc_b"]=le.fit_transform(data1["hsc_b"])
 data1["hsc_s"]=le.fit_transform(data1["hsc_s"])
 data1["degree_t"]=le.fit_transform(data1["degree_t"])
 data1["workex"]=le.fit_transform(data1["workex"])
 data1["specialisation"]=le.fit_transform(data1["specialisation"])
 data1["status"]=le.fit_transform(data1["status"])
 data1
 x=data1.iloc[:,:-1]
 x
 y=data1.iloc[:,-1]
 y
 from sklearn.model_selection import train_test_split
 x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
 from sklearn.linear_model import LogisticRegression
 model=LogisticRegression(solver="liblinear")
 model.fit(x_train,y_train)
 y_pred=model.predict(x_test)
 y_pred,x_test
 from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
 accuracy=accuracy_score(y_test,y_pred)
 confusion=confusion_matrix(y_test,y_pred)
 cr=classification_report(y_test,y_pred)
 print("accuracy score:",accuracy)
 print("\nconfusion matrix:\n",confusion)
 print("\nclassification report:\n",cr)
 from sklearn import metrics
 cm_display=metrics.ConfusionMatrixDisplay(confusion_matrix=confusion)
 cm_display.plot()

```

## Output:
![image](https://github.com/user-attachments/assets/7608692b-f448-481a-b370-bed9af1219db)
![image](https://github.com/user-attachments/assets/99227c9e-5f96-48cb-aee6-ad635fa206ec)
![image](https://github.com/user-attachments/assets/d7fd6e93-e456-491a-88be-447cd5e4024d)
![image](https://github.com/user-attachments/assets/922590e7-d86c-47df-a165-bdc6f8528c3a)
![image](https://github.com/user-attachments/assets/71b344cc-e7c6-4e8e-acaf-47a29b1a4d82)
![image](https://github.com/user-attachments/assets/329c7a44-5399-4456-b656-084cb94ec19b)
![image](https://github.com/user-attachments/assets/e32438ee-45cb-479d-8a7b-c07540e3ce0f)
![image](https://github.com/user-attachments/assets/a86520fa-46da-4b38-84e2-ecb620cdb8ad)
![image](https://github.com/user-attachments/assets/b26c69ac-ae1d-4706-8105-9b9ee9e52ecf)
![image](https://github.com/user-attachments/assets/69c3bce6-bafb-4a3e-9683-abd04966218a)
![image](https://github.com/user-attachments/assets/4d7b7348-dfc7-4dab-a6cc-8f8bdce296c2)
![image](https://github.com/user-attachments/assets/5467dcfc-14e8-417f-942b-fb44bc10b496)














## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
