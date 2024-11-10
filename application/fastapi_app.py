from fastapi import FastAPI, UploadFile
import pickle
from pydantic import BaseModel
from io import StringIO
import pandas as pd


app=FastAPI()
pickle_in = open("notebook/classifier.pkl","rb")
classifier=pickle.load(pickle_in)

class Data(BaseModel):
    variance: float
    skewness: float
    curtosis: float
    entropy: float

@app.get('/')
def welcome():
    return "Welcome All"

@app.post('/predict/')
async def pred_note_authentication(req:Data):
    prediction = classifier.predict([[req.variance, req.skewness, req.curtosis, req.entropy]])
    return f"The predicted value is: {str(prediction)}"

@app.post('/predict_file/')
async def predict_file(file:UploadFile):
    contents = await file.read()  # Read the file content
    str_content = contents.decode('utf-8')  # Convert bytes to string
    data = StringIO(str_content)  # Convert string to StringIO for pandas to read
    df_test = pd.read_csv(data)  # Read the CSV into a pandas DataFrame
    prediction = classifier.predict(df_test)
    return f"The predicted value is: {str(list(prediction))}"

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)