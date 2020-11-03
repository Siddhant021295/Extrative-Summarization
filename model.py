from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Dropout
def model ():
    model = Sequential([
        Conv2D(32, kernel_size=3, activation='relu', input_shape=(28, 28, 3)),
        BatchNormalization(),
        Conv2D(32, kernel_size=3, activation='relu'),
        BatchNormalization(),
        Conv2D(32, kernel_size=5, strides=2, padding='same', activation='relu'),
        BatchNormalization(),
        Dropout(0.4),
        Conv2D(64, kernel_size=3, activation='relu'),
        BatchNormalization(),
        Conv2D(64, kernel_size=3, activation='relu'),
        BatchNormalization(),
        Conv2D(64, kernel_size = 5, strides=2, padding='same', activation='relu'),
        BatchNormalization(),
        Dropout(0.4),
        Conv2D(128, kernel_size = 4, activation='relu'),
        BatchNormalization(),
        Flatten(),
        Dropout(0.4),
        Dense(10, activation='softmax')
    ])
    return model

# def model ():
#     model = Sequential([
#         Conv2D(64, (3,3), activation='relu', input_shape=(28, 28, 3)),
#         MaxPooling2D(2,2),
#         Conv2D(64,(3,3),activation='relu'),
#         MaxPooling2D(2,2),
#         Flatten(),
#         Dense(128,activation='relu'),
#         Dense(10,activation='softmax')
#     ])
#     return model