---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.2'
      jupytext_version: 1.6.0
  kernelspec:
    display_name: Python 3.8.3 64-bit
    language: python
    name: python38364bitc826dcc705f3421fa626a7aa062cf49f
---

<!-- #region -->
Nir Lotan's "Cheat Sheet" for machine learning, python, etc.
This notebook in no way contains a complete knowledge, but can be used as reference for many common actions.



# Jupyter Notebooks
## General tips

Use Jupytext extension in order to sync your notebook with a script or MD file for easy compare and merge: https://jupytext.readthedocs.io
        
Use nbdiff to diff your notebook from a github notebook: https://github.com/jupyter/nbdime
<!-- #endregion -->

# Basic Python tricks


## Loops


### iterator with counter

```python
presidents = ["Washington", "Adams", "Jefferson", "Madison", "Monroe", "Adams", "Jackson"]
for num, name in enumerate(presidents, start=1):
    print("President {}: {}".format(num, name))
```

### zip iterator

```python
colors = ["red", "green", "blue", "purple"]
ratios = [0.2, 0.3, 0.1, 0.4]
for color, ratio in zip(colors, ratios):
    print(f"{ratio*100}% {color}")
```

## Convert string list to list

```python
import ast
str = "['17915334', '17929027', '18182384', '20322929']"
ast.literal_eval(str)
```

# Pandas
## Pandas basics


### Read from and save to file

```python
import pandas as pd

df = pd.read_csv("indexes.csv")
df.to_csv("indexes.csv")
```

###  Get rid of unnamed columns

```python
import pandas as pd

df = pd.DataFrame(columns=['Unnamed: 0', 'unnamed','my_column'])
df.drop(df.columns[df.columns.str.contains('unnamed',case = False)],axis = 1, inplace = True)
```

### Drop duplicates

```python
df = df.drop_duplicates(subset=None, keep='first', inplace=False)
```

### Concatenate dataframes

```python
df1 = pd.DataFrame()
df2 = pd.DataFrame()

frames = [df1,df2]
new_df = pd.concat(frames, sort=False)
```

### Merge dataframes

```python
df1 = pd.DataFrame(columns=['common_name'])
df2 = pd.DataFrame(columns=['common_name'])


new_df = pd.merge(df1, df2, how='outer', indicator=True)
```

## Profiling

```python
import pandas_profiling
pandas_profiling.ProfileReport(df)
```

## Qgrid


```python
import pandas as pd
import qgrid

df = pd.DataFrame()

qgrid_widget = qgrid.show_grid(df,show_toolbar=True,grid_options={'forceFitColumns': False, 'defaultColumnWidth': 100})
qgrid_widget
```

### After modifying Qgrid - make sure to update the dataframe

```python
new_df = qgrid_widget.get_changed_df()
```

# OS
## Open all files in folder

```python
import os

origin_path = '/Users/nlotan/Documents'
for filename in os.listdir(origin_path):
    if filename.endswith(".txt"):
        print ('Do Something, e.g. print filename: ' + filename)

```

##  Open all files in folder and subfolders

```python
import os

root_folder = os.getcwd()

for path, subdirs, files in os.walk(root_folder):
    for name in files:
        print (os.path.join(path, name))
```

## Go over all lines in file

```python
file = open(filename,'r')
for line in file:
    print (line)
```

## Rename file

```python
os.rename(src, dst)
```

# Jupyter notebook widgets


## Dropdown text widget for selection and using the value

```python
import ipywidgets as widgets

list_of_values = ['Nir','Lotan']

w = widgets.Dropdown(
    options=list_of_values,
    value=list_of_values[0],
    description='Type:', layout={'width': 'max-content'},
    disabled=False,
    
)

display(w)
```

### Get the selected value like this

```python
w.value
```

# Visualization and Graphs


## Seaborn Histogram

```python
import seaborn as sns
import numpy as np
df = pd.DataFrame(np.random.randint(0,100,size=(100, 4)), columns=list('ABCD'))

#%matplotlib inline
sns.distplot(df['A'].dropna())
```

## Seaborn Heatmap
ref: https://github.com/mGalarnyk/Python_Tutorials/blob/master/Request/Heat%20Maps%20using%20Matplotlib%20and%20Seaborn.ipynb

```python
import seaborn as sns
import pandas as pd


helix = pd.read_csv('helix_parameters.csv')
couple_columns = helix[['Energy','helix 2 phase', 'helix1 phase']]
phase_1_2 = couple_columns.groupby(['helix1 phase', 'helix 2 phase']).mean()
phase_1_2.pivot('helix1 phase', 'helix 2 phase')['Energy'].head()

plt.figure(figsize=(9,9))
pivot_table = phase_1_2.pivot('helix1 phase', 'helix 2 phase','Energy')
plt.xlabel('helix 2 phase', size = 15)
plt.ylabel('helix1 phase', size = 15)
plt.title('Energy from Helix Phase Angles', size = 15)
sns.heatmap(pivot_table, annot=True, fmt=".1f", linewidths=.5, square = True, cmap = 'Blues_r');
```

# Configurations


## Work with toml configuration files


For this example toml config file, use the below code to use it:

```python
'''
toml config file content:
[servers]

  [servers.alpha]
  ip = "10.0.0.1"

  [servers.beta]
  ip = "10.0.0.2"

'''
```

```python
import toml
data = toml.load("data.toml")
data['servers']['alpha']['ip']
```

# Parser & Logger
## Parser

```python
from optparse import OptionParser

parser = OptionParser()
parser.add_option("-f", "--file", dest="filename",
                  help="write report to FILE", default="default_file.txt", metavar="FILE")

(options, args) = parser.parse_args()

print (options.filename)

```

## Better parser - Click

```python
import click

@click.command()
@click.option('--count', default=1, help='Number of greetings.')
@click.option('--name', prompt='Your name',
              help='The person to greet.')
def hello(count, name):
    """Simple program that greets NAME for a total of COUNT times."""
    for x in range(count):
        click.echo('Hello %s!' % name)

if __name__ == '__main__':
    hello()
```

## Logger

```python
import logging
from importlib import reload

#Initialization
reload(logging)
logging.basicConfig(filename='my_log.log', filemode='w', format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p',level=logging.DEBUG)

#The actual logging
logging.debug('Completed user#: '+ str(i) + ' uid:' + str(uid))
```

# Multi Threading

```python
from threading import Thread
import _thread

def my_func(instance_id):
    print(instance_id)

for threadnum in range(7):
    t = Thread(target=my_func,args=[threadnum])
    t.start()
```

# ML Flow

```python
import mlflow

# set mlflow environment for tracking experiments
mlflow.set_tracking_uri("/Users/nlotan/mlruns/")


exp_name = "Fizz Experiment"

try:
    # only the first time of running this experiment
    exp = mlflow.create_experiment(exp_name)
    mlflow.start_run(experiment_id=exp)
except:
    #experiment already exists, Assume the experiment number (taken from MLFlow UI)
    mlflow.set_experiment(exp_name)

Value = 7
mlflow.log_param("Name", Value)
mlflow.log_param("Another Name", "Text description")

mectric_value = 1
mlflow.log_metric("Accuray", mectric_value)

plt.savefig('confusion.png')
mlflow.log_artifact("confusion.png")

mlflow.end_run()
```

# Save and load models

```python
# Save Model Using Pickle
import pandas
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
import pickle
#url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
url = "dataset.csv"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = pandas.read_csv(url, names=names)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
test_size = 0.33
seed = 7
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=test_size, random_state=seed)
# Fit the model on training set
model = LogisticRegression()
model.fit(X_train, Y_train)
# save the model to disk
filename = 'finalized_model.sav'
pickle.dump(model, open(filename, 'wb'))
 
# some time later...
 
# load the model from disk
loaded_model = pickle.load(open(filename, 'rb'))
result = loaded_model.score(X_test, Y_test)
print(result)
```

# Text analysis


## Sentiment Analysis

```python
from empath import Empath
lexicon = Empath()
```

```python
lexicon.analyze("Oh, when Karens take a walk with their dogs off leash in the famous Bramble in NY’s Central Park, where it is clearly posted on signs that dogs MUST be leashed at all times, and someone like my brother (an avid birder) politely asks her to put her dog on the leash, she threat him.", normalize=True)
# => {'violence': 0.2}
```

## Easy BERT sentence embeddings

```python
#this line for the initial download
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import sister
sentence_embedding = sister.MeanEmbedding(lang="en")

sentence = "I am a dog."
vector = sentence_embedding(sentence)
```

# REST API
Example of the most basic REST API

```python
from flask import Flask
from flask import request

app = Flask(__name__)

@app.route('/hello/', methods=['GET', 'POST'])
def welcome():
    
    
    page = request.args.get('page', default = 1, type = int)
    name = request.args.get('name', default = '*', type = str)
    age = request.args.get('age', default = '*', type = str)
    return f"Hello {name}, you are {age} years old! "
    
app.run(host='0.0.0.0', port=105)
```

# Keras


## Sequencial Model

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
import numpy
seed = 7
numpy.random.seed(seed)


X = np.array([800, 720, 690, 250, 450, 325, 725, 777, 444, 692 , 300])
y = np.array([1,   1,   1,   0,   0,   0,   1,   1,   0,   1,    0  ])


model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(400, input_dim=1, activation='relu'))
model.add(tf.keras.layers.Dense(500, activation='relu'))
model.add(tf.keras.layers.Dense(20, activation='relu'))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy', tf.keras.metrics.AUC()]
             )
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=seed)
res = model.fit(X_train, y_train, epochs=100, batch_size=50, validation_split=0.1)


```

```python
for key in res.history:
    print(f"validation_{key}:\t\t{res.history[key][-1:][0]}")
    
_, accuracy, auc = model.evaluate(X_test, y_test)
print(f"test_accuracy:\t\t\t{accuracy}")
print(f"test_auc:\t\t\t{auc}")
```
