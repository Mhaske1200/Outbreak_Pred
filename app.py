import requests
from flask import Flask,request, url_for, redirect, render_template
import pickle
import numpy as np

app = Flask(__name__)

model=pickle.load(open('model.pkl','rb'))
label_encoder=pickle.load(open('label_encoder.pkl','rb'))

@app.route('/')
def hello_world():
    return render_template("index.html")

@app.route('/predict',methods=['POST','GET'])
def predict():
    float_features = [(x) for x in request.form.values()]
    final = [np.array(float_features)]
    print(final)
    city = final[0]
    city=str(city[0])
    print(city)
    url = 'https://api.openweathermap.org/data/2.5/weather?q={}&appid=3e02b66a6e63506b7c74c2aeeed8d43b&units=metric'.format(city)
    res = requests.get(url)
    data=res.json()
    print(data)
    # getting the main dict block
    main_data = data['main']
    wind = data['wind']
    # getting temperature
    temperature = main_data['temp']
    # getting the humidity
    humidity = main_data['humidity']
    tempmin = main_data['temp_min']
    tempmax = main_data['temp_max']
    # getting the pressure
    windspeed = wind['speed']
    pressure = main_data['pressure']
    # weather report
    report = data['weather']
    print(f"Temperature : {temperature}°C")
    print(f"Temperature Min : {tempmin}")
    print(f"Temperature Max : {tempmax}")
    print(f"Humidity : {humidity}")
    print(f"Pressure : {pressure}")
    print(f"Wind Speed : {windspeed}")
    print(f"Weather Report : {report[0]['description']}")
    inputt = []
    inputt.append(tempmax)
    inputt.append(tempmin)
    inputt.append(humidity)
    final = [np.array(inputt)]
    print(final)
    prediction = model.predict_proba(final)
    print(prediction)
    output = '{0:.{1}f}'.format(prediction[0][1], 2)
    print(output)
    if output>=str(0.5):
        return render_template('index.html',pred='BC tu Marnar\nProbability of Malaria Breeds occuring is {}\n '.format(output))
    else:
        return render_template('index.html',pred='Less chance of Malaria Outbreak\n Probability of Malaria Breeds occuring is {}\n '.format(output))


if __name__ == '__main__':
    app.run(debug=True)