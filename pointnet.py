import os
from utils import *

if __name__ == "__main__":
    list_size = 1024
    data_set = []
    label_set = []
    for folder_name in os.listdir('MATLAB/Point Cloud Dataset'):
        # print('Directory:', folder_name)
        for file_name in os.listdir('MATLAB/Point Cloud Dataset/' + folder_name):
            # print('File:', file_name)
            file = open('MATLAB/Point Cloud Dataset/' + folder_name + '/' + file_name)
            data_point = np.genfromtxt('MATLAB/Point Cloud Dataset/' + folder_name + '/' + file_name, delimiter=',').tolist()
            data_point += [[0, 0, 0, 0]] * (list_size - len(data_point))
            label_set.append(int(file_name[0]))
            data_set.append(data_point)
    label_set = np.array(label_set)
    data_set = np.array(data_set)

# Data Split for training and testing
train_dataset = np.random.rand(1500, 1024, 4)
test_dataset = np.random.rand(300, 1024, 4)

# Model Parameters
NUM_POINTS = 1024
NUM_CLASSES = 6

# Inputs
inputs = keras.Input(shape=(NUM_POINTS, 4))

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
# model.fit(train_dataset, epochs=20, validation_data=test_dataset)

# Save the trained model
