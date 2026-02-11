# Import necessary lib
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from datetime import date
from sklearn.model_selection import train_test_split

clim = pd.read_csv('wk08/climate_2009.csv', index_col = 0)

clim = clim.reset_index()
print(clim.head())

# Replace column name
df = clim[['Date Time','T (degC)']].rename(columns = {'T (degC)':'T','Date Time':'datetime'})
print(df.head())

df.plot(figsize = (15,5))
#plt.show()

df['datetime'] = pd.to_datetime(df['datetime'], format='%d.%m.%Y %H:%M:%S')
print(type(df['datetime'][0]))
print(df.head(10))

# Taking each 6th record as we need hourly data, so we can ignore every other record (which are on 10 min level)
# We can also take mean/median of each consecutive 6 records if that is a business requirement, but right now it is not.
df_hour_lvl = df[5::6].reset_index().drop('index', axis=1)
print(df_hour_lvl.shape)
print(df_hour_lvl.head())

## Check the data distribution ##
plt.figure(figsize = (14,5))
sns.distplot(df_hour_lvl['T'])
plt.show()

## Train - Test split ##
# we cannot simply provide a sequence of data points to an LSTM model for training and testing, 
# the input X sequence must be a 2D array [no of records, n_input], while the input y sequence must be a 1D array
def Sequential_Input_LSTM(df, input_sequence):
    df_np = df.to_numpy()
    x = []
    y = []
    
    for i in range(len(df_np) - input_sequence):
        row = [a for a in df_np[i:i + input_sequence]]
        x.append(row)
        label = df_np[i + input_sequence]
        y.append(label)
        
    return np.array(x), np.array(y)

n_input = 10      # number of historical inputs to be used for forecasting the future time series 
df_min_model_data = df_hour_lvl['T']
x, y = Sequential_Input_LSTM(df_min_model_data, n_input)
print(x.shape, y.shape)

# size of train and val data
train_size = int(len(x) * 0.8)
val_size = int(len(x) * 0.1)

# Training data
#x_train, y_train = x[:1400], y[:1400]
x_train, y_train = x[:train_size], y[:train_size]

# Validation data
#x_val, y_val = x[1400:1600], y[1400:1600]
x_val, y_val = x[train_size:train_size+val_size], y[train_size:train_size+val_size]

# Test data
#x_test, y_test = x[1600:], y[1600:]
x_test, y_test = x[train_size+val_size:], y[train_size+val_size:]

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42, shuffle=True  #stratify=y
)

print("Train:", x_train.shape, y_train.shape)
print("Val:", x_val.shape, y_val.shape)
print("Test:", x_test.shape, y_test.shape)

### DAYS ###
print('Temp :')
print()
print(f'Total days      : {len(df_hour_lvl)/24}')
print(f'Training days   : {len(x_train)/24}')
print(f'Validation days : {len(x_val)/24}')
print(f'Testing days    : {len(x_test)/24}')

# -------------------------------------------------------------#
### Creating the LSTM Model ###
from tensorflow.keras.models import Sequential, save_model, load_model
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.optimizers import Adam 

n_features = 1      # number of input variables used for forecast (here, only 1 i.e. temperature)

model1 = Sequential()

model1.add(InputLayer((n_input,n_features)))
model1.add(LSTM(100, return_sequences = True))     
model1.add(LSTM(100, return_sequences = True))
model1.add(LSTM(50))
model1.add(Dense(8, activation = 'relu'))
model1.add(Dense(1, activation = 'linear'))

model1.summary()

early_stop = EarlyStopping(monitor = 'val_loss', patience = 2)
model1.compile(loss = MeanSquaredError(), optimizer = Adam(learning_rate = 0.0001), metrics = [RootMeanSquaredError()])
model1.fit(x_train, y_train, validation_data = (x_val, y_val), epochs = 50, callbacks = [early_stop])

losses_df1 = pd.DataFrame(model1.history.history)
print(losses_df1)

losses_df1.plot(figsize = (10,6))
plt.show()

scores = model1.evaluate(x_test, y_test, verbose=0)
print("%s: %.2f%%" % (model1.metrics_names[0], scores[0]*100))
print("%s: %.2f%%" % ('RootMeanSquaredError', scores[1]*100))

### Predict on Test data ###
test_predictions1 = model1.predict(x_test).flatten()
test_predictions1

print(len(x_test), len(test_predictions1))

## Check the dimension of the output (should be 1 for time series forecast)
print(test_predictions1.ndim)

x_test_list = []
for i in range(len(x_test)):
    x_test_list.append(x_test[i][0])
print(len(x_test_list))

test_predictions_df1 = pd.DataFrame({'x_test':list(x_test_list), 'LSTM Prediction':list(test_predictions1)})
print(len(test_predictions_df1))

print(test_predictions_df1.head())


### LSTM temp forecast on complete Test data (6 Days)
test_predictions_df1.plot(figsize = (15,6))
plt.show()
