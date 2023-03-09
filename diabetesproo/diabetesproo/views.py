from django.shortcuts import render
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def index(request):
    return render(request,'index.html')
def predict(request):
    return render(request,'predict.html')
def result(request):
    data = pd.read_csv(r"C:\Users\91825\OneDrive\Desktop\Diabeticpro\diabetesproo\dataset\diabetes.csv")
    x = data.drop("Outcome", axis=1)
    y = data['Outcome']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    model = LogisticRegression()
    model.fit(x_train, y_train)
    yal1=float(request.GET['n1'])
    yal2=float(request.GET['n2'])
    yal3=float(request.GET['n3'])
    yal4=float(request.GET['n4'])
    yal5=float(request.GET['n5'])
    yal6=float(request.GET['n6'])
    yal7=float(request.GET['n7'])
    yal8=float(request.GET['n8'])

    pred = model.predict([[yal1, yal2, yal3, yal4, yal5, yal6, yal7, yal8]])

    result1 = ""
    if pred==[1]:
        result1="Positive"
    else:
        result1="Negative"

    return render(request, "predict.html", {"result2":result1})