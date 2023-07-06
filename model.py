import requests
city='Pune'

api_key1 = '3e02b66a6e63506b7c74c2aeeed8d43b'
# url= 'http://api.openweathermap.org/data/2.5/weather?q={}&appid={}&units=metric'.format(city,api_key1)
url= 'https://api.openweathermap.org/data/2.5/weather?q={}&appid={}'.format(city,api_key1)

res= requests.get(url)
data=res.json()

latitude = data['coord']['lat']
longitude = data['coord']['lon']
print('latitude :',latitude)
print('longitude :',longitude)
   # getting the main dict block
main = data['main']
wind = data['wind']
  # getting temperature
temperature = main['temp']
  # getting the humidity
humidity = main['humidity']
tempmin=main['temp_min']
tempmax=main['temp_max']
  # getting the pressure
windspeed=wind['speed']
pressure = main['pressure']
  # weather report
report = data['weather']
print(f"Temperature : {temperature}°C")
print(f"Temperature Min : {tempmin}")
print(f"Temperature Max : {tempmax}")
print(f"Humidity : {humidity}")
print(f"Pressure : {pressure}")
print(f"Wind Speed : {windspeed}")
print(f"Weather Report : {report[0]['description']}")

#---------------------------------------------------------------------------------

dataset = 'outbreak_detect.csv'

import pandas as pd
import numpy as np
import pickle

df = pd.read_csv(dataset)

from sklearn.preprocessing import LabelEncoder

outbreak=df['Outbreak']

label_encoder = LabelEncoder()
df['Outbreak'] = label_encoder.fit_transform(outbreak)


df = df.drop('Positive',axis=1)
df = df.drop('pf',axis=1)
df = df.drop('Rainfall',axis=1)

x = df.drop(columns='Outbreak')
y = df['Outbreak']

from sklearn.model_selection import train_test_split
Xtrain , Xtest , Ytrain , Ytest = train_test_split(x,y , test_size=0.2 , random_state=0)

from sklearn.neighbors import KNeighborsClassifier
knn_clf=KNeighborsClassifier(n_neighbors=3)
knn_clf.fit(Xtrain,Ytrain)

test_accuracy=knn_clf.score(Xtest,Ytest)
print(test_accuracy)

train_accuracy=knn_clf.score(Xtrain,Ytrain)
print(train_accuracy)

overall_accuracy= knn_clf.score(x,y)
print(overall_accuracy)

inputt=[]
inputt.append(tempmax)
inputt.append(tempmin)
inputt.append(humidity)
final = [np.array(inputt)]
print(final)
prediction=knn_clf.predict_proba(final)
output = '{0:.{1}f}'.format(prediction[0][1], 2)
print(output)
if output>=str(0.5):
  print(1)
else:
  print(0)


pickle.dump(knn_clf,open('model.pkl','wb'))
pickle.dump(label_encoder,open('label_encoder.pkl','wb'))
model=pickle.load(open('model.pkl','rb'))
label=pickle.load(open('label_encoder.pkl','rb'))
print("Sucess loaded")