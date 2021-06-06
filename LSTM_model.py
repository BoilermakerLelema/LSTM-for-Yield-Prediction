import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def TS_nonTS_LSTM_model(x_TS_shape, x_nonTS_shape):
	inn = layers.Input(shape=(x_TS_shape[1], x_TS_shape[2]),name='TS_input')
	lstm_layer1 = layers.LSTM(128, return_sequences=True)(inn)
	lstm_layer1 = layers.Dropout(0.5)(lstm_layer1)

	lstm_layer2 = layers.LSTM(64, return_sequences=True)(lstm_layer1)
	lstm_layer2 = layers.Dropout(0.5)(lstm_layer2)

	lstm_layer3 = layers.LSTM(32, return_sequences=False)(lstm_layer2)
	lstm_layer3 = layers.Dropout(0.5)(lstm_layer3)

	inn2 = layers.Input(shape=(x_nonTS_shape[1]), name='nonTS_input')

	#merged layer
	merged_layer = layers.concatenate([lstm_layer3, inn2])
	hidden_layer = layers.Dense(64, activation='relu')(merged_layer) # new
	hidden_layer = layers.Dropout(0.5)(hidden_layer)
	
	#hidden_layer2 = layers.Dense(32, activation='relu')(hidden_layer) # new
	#hidden_layer2 = layers.Dropout(0.5)(hidden_layer2)
	
	outt_TS = layers.Dense(1, name='output',kernel_initializer='normal')(hidden_layer)
   
	model_both = keras.Model(inputs=[inn, inn2], outputs=outt_TS)
	model_both.compile(optimizer=keras.optimizers.Adam(lr=1e-3),
             loss='mean_squared_error',
             metrics=['mae'])

	return model_both
