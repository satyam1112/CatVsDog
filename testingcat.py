from keras.models import load_model
# Load the trained model
model = load_model("CatVsDogModel50.h5")
from keras.preprocessing.image import ImageDataGenerator

test_datagen = ImageDataGenerator(rescale=1./255)


val_gen = test_datagen.flow_from_directory(r'D:\CatvsDogs\test', 
                                          class_mode='binary',
                                          batch_size = 32,
                                          target_size = (128,128),
                                          shuffle = False
                                         )



result = model.predict(val_gen,batch_size = 32,verbose = 0)

# y_pred = np.argmax(result, axis = 1)

# y_true = test_generator.labels

# Evaluvate
loss,acc = model.evaluate(val_gen, batch_size = 32, verbose = 0)

print('The accuracy of the model for testing data is:',acc*100)
print('The Loss of the model for testing data is:',loss)