from flask import Flask, request, render_template, jsonify
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from src.pipline.predict_pipeline import PredictPipeline, CustomData
application= Flask(__name__)
app=application
# Define the route for the home page
@app.route('/')
def index():
    return render_template('index.html')    
@app.route('/predictdata', methods=['GET','POST'])
def predictdata():
    # Add your prediction logic here
    if request.method == 'GET':
        return render_template('home.html')
    else:
        data=CustomData(
            gender=request.form.get('gender'),
            race_ethnicity=request.form.get('ethnicity'),
            lunch=request.form.get('lunch'),
            parental_level_of_education=request.form.get('parental_level_of_education'), 
            test_preparation_course=request.form.get('test_preparation_course'),   
            writing_score=float(request.form.get('writing_score')),
            reading_score=float(request.form.get('reading_score'))
                    
        )
        pred_df=data.get_data_as_data_frame()
        print(pred_df)
        predict_pipeline=PredictPipeline()
        results=predict_pipeline.predict(pred_df)
        return render_template('home.html', results=results[0])
if __name__ == "__main__":
    app.run(host='0.0.0.0',debug=True)  # Set debug=True for development
