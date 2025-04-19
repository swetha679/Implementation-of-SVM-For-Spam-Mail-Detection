# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM for Spam Mail Detection.

## Equipment Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the necessary Python packages using import statements.
2. Read the given CSV file using the read_csv() method and print the number of contents to be displayed using df.head().
3. Split the dataset using train_test_split.
4. Calculate Y_Pred and accuracy.
5. Print all the outputs.
6. End the Program.

## Program:
```

Program to implement the SVM for Spam Mail Detection..
Developed by: B R SWETHA NIVASINI 
Register Number:  212224040345

```
```
import chardet
file='spam.csv'
with open (file,'rb') as rawdata:
    result = chardet.detect(rawdata.read(100000))
result

import pandas as pd
data=pd.read_csv("spam.csv",encoding='windows-1252')

data.head()

data.info()

data.isnull().sum()

x=data["v1"].values
y=data["v2"].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()

x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)

from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
y_pred

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
```

## Output:
![SVM For Spam Mail Detection](sam.png)

# Encoding
![Screenshot 2025-04-19 171753](https://github.com/user-attachments/assets/f486db4c-29d4-4d62-b575-f148e8af6397)

# Head

![Screenshot 2025-04-19 171802](https://github.com/user-attachments/assets/3eef4370-5117-4072-8c0a-309453a0bd28)

# Info()

![Screenshot 2025-04-19 171811](https://github.com/user-attachments/assets/8a0987d3-d6fa-4eeb-bfb0-4eb40f6aac09)


# isnull().sum()

![Screenshot 2025-04-19 171820](https://github.com/user-attachments/assets/c607c5bd-4d9e-441e-be25-4147a8292e68)

# Prediction of y
![Screenshot 2025-04-19 171828](https://github.com/user-attachments/assets/92f5f39d-db82-4e33-9c27-e9c7851bc8fa)








## Result:
Thus, the program to implement the SVM for Spam Mail Detection is written and verified using Python programming.
