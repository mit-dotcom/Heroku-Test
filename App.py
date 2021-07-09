#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os 
from flask import send_from_directory
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
import pickle
import logging

app = Flask(__name__)
model = pd.read_pickle('model.pkl')

@app.route('/')
def home():
    return render_template('forecast1.html')

@app.route('/favicon.ico') 
def favicon(): 
    return send_from_directory(os.path.join(app.root_path, 'static'), 'favicon.ico', mimetype='image/vnd.microsoft.icon')

@app.route('/predict',methods=['POST'])
def predict():

    days = [y for y in request.form.values()]
    day = int(days[0])
    hour = 24 * (day-1)
                                                                        
    y_hat = pd.DataFrame(model.predict(start=4433,end=4433+hour))
    y_hat = (y_hat).to_string(index=False)
    pred = y_hat.to_html()
    text_file = open("templates/pred.html", "w")
    text_file.write(pred)
    text_file.close()
    return render_template('forecast1.html', output='Forecast for next {} days {}'.format(day,y_hat))
    
if __name__ == "__main__":
    app.run(debug=False)
    
# In[ ]:




