import matplotlib.pyplot as plt
import json
import pandas as pd

df_s = pd.read_json('sequential.json')
df_mp = pd.read_json('openMP.json')
df_cuda = pd.read_json('cuda.json')

training_seconds = plt.gca()
df_s.plot(kind='line',x='number of epochs',y='Training duration in seconds',ax=training_seconds)
df_mp.plot(kind='line',x='number of epochs',y='Training duration in seconds',color='red',ax=training_seconds)
df_cuda.plot(kind='line',x='number of epochs',y='Training duration in seconds',color='green',ax=training_seconds)
training_seconds.set_xlabel("number of epochs")
training_seconds.set_ylabel("training duration on seconds")
training_seconds.legend(['sequential', 'openMP','cuda'])

plt.show()
fig = training_seconds.get_figure()
fig.savefig("duration.png")

accuracy = plt.gca()
df_s.plot(kind='line',x='number of epochs',y='Accuracy',ax=accuracy)
df_mp.plot(kind='line',x='number of epochs',y='Accuracy',color='red',ax=accuracy)
df_cuda.plot(kind='line',x='number of epochs',y='Accuracy',color='green',ax=accuracy)
accuracy.set_xlabel("number of epochs")
accuracy.set_ylabel("Accuracy")
accuracy.legend(['sequential', 'openMP','cuda'])

plt.show()
fig = accuracy.get_figure()
fig.savefig("Accuracy.png")
