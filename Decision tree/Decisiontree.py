import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import seaborn as sns

df=pd.read_csv("D:\Projects\loan_default\Dataset\Loan_default.csv")
print(df.shape)
print(df.head())
print (df.info())
print(df.describe())

#check the missing value and store the missing value columns in a variable
print(df.isnull().sum())
missing= df.columns[df.isnull().any()]
print(missing)

print(df.duplicated().sum())

##Checking outliers
# num_cols = df.select_dtypes(include=['int64', 'float64']).columns
# for col in num_cols:
#     plt.figure(figsize=(6,3))
#     sns.boxplot(x=df[col])
#     plt.title(f"Boxplot for {col}")
#     plt.show()


# handle missing values for int, float, object dtypes
num_col= df.select_dtypes(include=["int64", "float64"]).columns
df[num_col]=df[num_col].fillna(df[num_col].median())
#print(df.isnull().sum())


ob_col= df.select_dtypes(include="object").columns
df[ob_col]=df[ob_col].fillna(df[ob_col].mode().iloc==0)
print(df.isnull().sum())

# LabelEncoding
for col in ob_col:
    le = LabelEncoder()
    df[col]= df[col].astype(str)
    df[col] = le.fit_transform(df[col])
print(df.head())


## optional steps for outliers chcking
# from sklearn.preprocessing import StandardScaler
# scaler = StandardScaler()

# scaled_cols = ['loan_amount', 'rate_of_interest', 'income', 'Credit_Score']
# df[scaled_cols] = scaler.fit_transform(df[scaled_cols])


# Feature split and target.
print(df.columns)
x=df.drop('Status', axis=1) #select all the columns except output
y=df['Status'] #select only output (target variable)

# split into train(80%) and test(20%)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

# model training
model=RandomForestClassifier(random_state=100)
model.fit(x_train,y_train)

predict=model.predict(x_test)

print("Accuracy:",accuracy_score(y_test,predict))
print("\nConfusion matrix: \n",confusion_matrix(y_test,predict))
print("\nclaasification report: \n", classification_report(y_test,predict))


#model training
model=DecisionTreeClassifier(random_state=50)
model.fit(x_train,y_train)
dt_predict=model.predict(x_test)
print(accuracy_score(y_test,dt_predict))

s= plt.figure(figsize=(15,10))
chart= tree.plot_tree(model,filled=True)
plt.show()



#pre pruning
model1=DecisionTreeClassifier(
    random_state=50,
    max_depth=5,
    min_samples_split=50,
    min_samples_leaf=15
)
model1.fit(x_train,y_train)
model_pre= model1.predict(x_test)
print("accuracy",accuracy_score(y_test,model_pre))

plt.figure(figsize=(15,10))
tree.plot_tree(model1,filled=True)
plt.show()

#post pruning
model2=DecisionTreeClassifier()

