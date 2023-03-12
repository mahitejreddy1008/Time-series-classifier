#import libraries needed
import os
import pandas as pd
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from csv import DictReader
from FeatureEngineeing import FE
import csv
from Models import Model
#using FastAPI for backend
app = FastAPI()
#basic functional UI for taking new csv fikle as input for prediction
@app.get("/")
async def read_form():
    html = """
    <html>
    <head>
    <title> Time Series Model </title>
    <body bgcolor = #90A9E8 >
        <center> <h1><marquee> <u>DataGenie Hackathon </u></marquee></h1></center>
        <center><h1> Time Series Classification Model </h1></center>
        
        <h2> Upload CSV File </h2>
        <strong><h3> Important Points to be Remembered before uploading the file: </h3></strong>
        <ol>
        <li>The Format of data file should be CSV</li>
        <li>The Data File Should have only 2 columns </li>
        <li>File Columns name should only be [point_timestamp,point_value]</li>
        <li> Date Format should be YYYY-MM-DD in Data File uploaded </li>
        </ol> 
        <form action="/uploadfile/" enctype="multipart/form-data" method="post">
        <input name="file" type="file">
        <p>Type Of Data[typeData]:</p>
        <p>0-> Daily</p>
        <p>1->Month</p>
        <p>2->Weakly</p> 
        <p>3-> Hourly</p>
        <input name="type" type="text" placeholder="Enter your file type" >
        <h2>Enter the Date that should be Forecasted: </h2>
        <input name="date" type="text" placeholder="Enter Date to be forecasted" >
        <input type="submit">
        <center><h5>Done By: Mahitej K 20PD14</h5></center>
        </form>
    </body>
    </head>
    </html>
    """
    return HTMLResponse(content=html, status_code=200)
#for uploading a file and store it in a local storage
@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile = File(...),type: str = Form(...),date: str = Form(...)):
    contents = await file.read()
    decoded = contents.decode('utf-8').splitlines()
    reader = csv.reader(decoded)
    data = [row for row in reader]
    for i in range(len(data)):
        data[i] = data[i][1:]
    cols = data[0]
    data = data[1:]
    df = pd.DataFrame(data,columns=cols)
    df.to_csv("result.csv",index=False)
    print("Success")
    best_model = FE(type)
    df.set_index("point_timestamp", inplace=True)
    df.index = pd.to_datetime(df.index)
    df.dropna(inplace=True)
    mape_best,forecaste  = Model(best_model,df,date)

#printing results by chosing best model and predicting future data
    html = f'''
        <html>
        <head>
        <title> Time Series Model </title>
        <body bgcolor = #90A9E8 >
        <center> <h1><marquee> <u>DataGenie Hackathon </u></marquee></h1></center>
        <center> <h2><u>Results From Time series classifier and the predicted value </u></h2></center>
        <h2><u>Results: </u></h2>
        <ol>
        <li> best model chose from time series classifier is: {best_model} </li>
        <li> Type of File: {type} </li>
        <li> MAPE(mean Absolute Percentage Error) Score of Test by Splitting in ratio 90:10 : {mape_best} </li>
        <li> Prediction for the date {date} is approximately equal to {forecaste}</li>
        <center><h5>Done By: Mahitej K 20PD14</h5></center>
        </ol>
        </body>
        </head>
        </html>
        '''
    return HTMLResponse(content = html,status_code=200)
