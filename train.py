#%%
from model_utils import save_result, create_model, load_train_test

#%%
DATA_PATH = 'data/'
INPUT_SHAPE = (32, 32, 3)
N_CLASS = 2
EPOCHS = 50
BATCH_SIZE = 10
OPTIMIZER = "adam"
LOSS = "categorical_crossentropy"

#%%
x_train, y_train, x_test, y_test = load_train_test(DATA_PATH)

x_train = x_train / 255.0
x_test = x_test / 255.0

print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)
#%%
prepare_inst = create_model(INPUT_SHAPE, N_CLASS, LOSS, OPTIMIZER, True)
nn_model =  prepare_inst.prepare_nn()

nn = nn_model.fit(x_train,y_train, batch_size = BATCH_SIZE, epochs = EPOCHS, verbose=1, 
                                        validation_data=(x_test, y_test), shuffle=True)

save_result(nn, nn_model, DATA_PATH)
