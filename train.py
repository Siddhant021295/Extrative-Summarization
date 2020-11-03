from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.optimizers import RMSprop
import model as m
train_dir ='./digit/trainingSet/'
validation_dir = './digit/testSet/'
train_datagen = ImageDataGenerator(
    rescale=1./255.,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range= 0.2,
    fill_mode='nearest',
)
validation_datagen = ImageDataGenerator(
    rescale=1./255.
)
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(28 ,28),
    batch_size= 375,
    class_mode= 'categorical',
)
validation_generator = train_datagen.flow_from_directory(
    validation_dir,
    target_size=(28 ,28),
    batch_size= 375,
    class_mode= 'categorical',
)
class myCallback(Callback):
    def on_epoch_end(self, epoch, logs={}):
        if (logs.get('acc')>0.99):
            print("\nAccuracy is too high")
            self.model.stop_traning =True

model = m.model()
callbacks = myCallback()
model.compile(loss ='categorical_crossentropy',
              optimizer=RMSprop(lr=0.001),
              metrics = ['acc'])
model.summary()
history = model.fit_generator(
    train_generator,
    steps_per_epoch=train_generator.samples//375,
    epochs =10,
    callbacks =[callbacks],
    verbose =1,
    validation_data =validation_generator,
    validation_steps=validation_generator.samples//375
)

model.save('./model_weights/weight')