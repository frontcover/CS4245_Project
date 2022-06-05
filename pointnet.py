import os
from utils import *

if __name__ == "__main__":
    for i in os.listdir('MATLAB/Point Cloud Dataset'):
        print('Directory:', i)
        for j in os.listdir('MATLAB/Point Cloud Dataset/' + i):
            print('File:', j)
            file = open('MATLAB/Point Cloud Dataset/' + i + '/' + j)
            data = np.genfromtxt('MATLAB/Point Cloud Dataset/' + i + '/' + j, delimiter=',')
            print(data)

# Model Parameters
NUM_POINTS = 1024
NUM_CLASSES = 6

# Inputs
inputs = keras.Input(shape=(NUM_POINTS, 3))

# Main Network (T-net removed)
# x = tnet(inputs, 3)
x = conv_bn(inputs, 32)
x = conv_bn(x, 32)
# x = tnet(x, 32)
x = conv_bn(x, 32)
x = conv_bn(x, 64)
x = conv_bn(x, 512)
x = layers.GlobalMaxPooling1D()(x)
x = dense_bn(x, 256)
x = layers.Dropout(0.3)(x)
x = dense_bn(x, 128)
x = layers.Dropout(0.3)(x)

# Output
outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)

# Build the model
model = keras.Model(inputs=inputs, outputs=outputs, name="pointnet")

# Model Training
model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    metrics=["sparse_categorical_accuracy"],
)
model.fit(train_dataset, epochs=20, validation_data=test_dataset)

# Save the trained model