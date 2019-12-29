import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
import tensorflow as tf
import numpy as np
import math
from keras import layers as l
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Reshape, Conv2DTranspose, BatchNormalization, Dropout, Flatten, concatenate
from keras.models import Model
from keras.utils import plot_model, get_file
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import re


data = 'RAVEN/TrialRightWrong/Trial'

# All the files we need
files = ['/DEI_trial_by_trial_Right.xlsx', 
         '/DEI_trial_by_trial_Wrong.xlsx', 
         '/esec_trial_by_trial_Right.xlsx',
         '/esec_trial_by_trial_Wrong.xlsx']
genreFile = 'RAVEN/Informação_género.txt'


# Initialize a dict for aggregate the fatigue values for each person and if answer right/wrong a question
values = { 'TRAINING': {}, 'TESTING': {} }
for file in files:
    
    # For each file
    f = pd.ExcelFile(data+file)

    # For each Train
    for sheet in f.sheet_names:
        
        df = pd.read_excel(f,  sheet_name=sheet)
        # Get the fatigue value for each person
        # In the 'unnamed:22' column is where person id is
        # Notice that if a person isn't in the 'right files', so he is in the 'wrong files' 
        category = 'TRAINING' if 'TRAINING' in sheet else 'TESTING' if 'TESTING' in sheet else None  
        if category and df.shape[0] > 0:
            dataValues = values[category]

            for index, row in df[df.columns].iterrows():
                # cell = [genre, course, fatigue_value, correct(1)/incorret(0)]
                *rest,id = row 
                cell = rest
                cell += [1 if (file=='/DEI_trial_by_trial_Right.xlsx' or file=='/esec_trial_by_trial_Right.xlsx') else 0]
                cell += [int("DEI" in id)] 
                if (id not in dataValues):
                    dataValues[id] = [cell]
                else:
                    dataValues[id] += [cell] 
                    
#Extract the genre from genreFile
with open(genreFile, "r") as f:
    for l in f:
        l = l.strip()
        if l=="" or '--' in l:continue
        id, genre, rest = l.split(' - ')
        
        if id[-2] == '_': id = id[:-1] + '0' + id[-1]
        for dataDict in [values['TRAINING'], values['TESTING']]: #get dicts train and test
            for i in range(len(dataDict[id])):
                dataDict[id][i].append(int(genre=='Masculino'))  

#print(values)
# For each person ( PERSON : list([genre, course, fatigue_value, correct/incorrect ]))        
items = { 'TRAINING': [], 'TESTING': [] }
for dataType in ['TRAINING','TESTING']:
    for v in values[dataType].values():
        items[dataType] += v
items['TRAINING'] = np.matrix(items['TRAINING'])
items['TESTING'] = np.matrix(items['TESTING'])

Itrain = items['TRAINING']   #introducing missing values by the average
col_mean = np.nanmean(Itrain, axis=0)
inds = np.where(np.isnan(Itrain))
Itrain[inds] = np.take(col_mean, inds[1])

scaler = MinMaxScaler()
scaler.fit(Itrain[:,:-2])

Itest = items['TESTING']   #fixing missing values by the average
col_mean = np.nanmean(Itest, axis=0)
inds = np.where(np.isnan(Itest))
Itest[inds] = np.take(col_mean, inds[1])

data_trainX = scaler.transform(Itrain[:,:-2])
data_trainY = Itrain[:,-2:]

data_testX = scaler.transform(Itest[:,:-2])
data_testY = Itest[:,-2:] 


#pca = PCA(n_components=2)
#pcaData = pca.fit_transform(data_trainX[:,2:-1])

#plt.scatter(pcaData[:,0], pcaData[:,1])

#plt.show()



inputs = Input(shape=(data_trainX.shape[1],))
x = Dense(32, activation='relu')(inputs)
x = Dense(8, activation='relu')(x)
x = Dense(8, activation='relu')(x)
outputs = Dense(data_trainY.shape[1], activation='sigmoid')(x)
model = Model(inputs=inputs, outputs=outputs)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])

plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=False)
model.fit(data_trainX, data_trainY, epochs=40, batch_size=8)
predicts = (model.predict(data_testX)>0.5)
print('-----')
genrePredicts = predicts[:,-1:]
genreTest = data_testY[:,-1:]
indnames = ['FEMALE','MALE']
print('genre accuracy:', (genrePredicts==genreTest).sum()/genrePredicts.shape[0])
for i in range(2):
    precision = np.multiply(genrePredicts==i,genreTest==i).sum()/((genrePredicts==i).sum())
    recall = np.multiply(genrePredicts==i,genreTest==i).sum()/((genreTest==i).sum())
    f1scor = 2*precision*recall/(precision+recall) if precision+recall else 0.0
    print('---')
    print(f"precision genre {indnames[i]}: {precision}")    
    print(f"reccal genre {indnames[i]}: {recall}")
    print(f"f1-scor genre {indnames[i]}: {f1scor}")

print('-----')
coursePredicts = predicts[:,-2:-1] #same as [:,-2] but in matrix form
courseTest = data_testY[:,-2:-1] #same as [:,-2] but in matrix form
indnames = ['ESEC','DEI']
print('course accuracy:', (coursePredicts==courseTest).sum()/coursePredicts.shape[0])
for i in range(2):
    divisor = (coursePredicts==i).sum()
    precision = np.multiply(coursePredicts==i,courseTest==i).sum()/divisor if divisor else 0
    recall = np.multiply(coursePredicts==i,courseTest==i).sum()/divisor if divisor else 0
    f1scor = 2*precision*recall/(precision+recall) if precision+recall else 0.0
    print('---')
    print(f"precision genre {indnames[i]}: {precision}")    
    print(f"reccal genre {indnames[i]}: {recall}")
    print(f"f1-scor genre {indnames[i]}: {f1scor}")
print('-----')

groupPredicts = predicts[:,-1:]+predicts[:,-2:-1]*2 #0,1,2,3
groupTest = data_testY[:,-1:]+data_testY[:,-2:-1]*2 #0,1,2,3
print('group accuracy:',(groupPredicts==groupTest).sum()/groupPredicts.shape[0])

indnames = ['ESEC AND FEMALE','ESEC AND MALE', 'DEI AND FEMALE', 'DEI AND MALE']
for i in range(4):
    divisor = (groupPredicts==i).sum()
    precision = np.multiply(groupPredicts==i,groupTest==i).sum()/divisor if divisor else 0
    recall = np.multiply(groupPredicts==i,groupTest==i).sum()/divisor if divisor else 0
    f1scor = 2*precision*recall/(precision+recall) if precision+recall else 0.0
    print('---')
    print(f"precision group {indnames[i]}: {precision}")    
    print(f"reccal group {indnames[i]}: {recall}")
    print(f"f1-scor group {indnames[i]}: {f1scor}")
    
