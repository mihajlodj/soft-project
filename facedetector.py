import cv2
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.models import Model
import tensorflow as tf
import json
import numpy as np
import os
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.layers import GlobalMaxPooling2D

HEIGHT = 120
WIDTH = 120

TRAIN_IMAGES = './depricated/aug_data-depricated/train/images'
TRAIN_LABELS = './depricated/aug_data-depricated/train/labels'
TEST_IMAGES = './depricated/aug_data-depricated/test/images'
TEST_LABELS = './depricated/aug_data-depricated/test/labels'
VAL_IMAGES = './depricated/aug_data-depricated/val/images'
VAL_LABELS = './depricated/aug_data-depricated/val/labels'

FACE_DETECTOR = './facedetector_v4.h5'

def build_model():
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(120, 120, 3))
    for layer in base_model.layers:
        layer.trainable = False

    # f1 = GlobalMaxPooling2D()(base_model)
    # x = Dense(2048, activation='relu')(f1)
    x = base_model.output
    x = Flatten()(x)
    x = Dense(2048, activation='relu')(x)
    x = Dropout(0.15)(x)
    # x = Dense(1024, activation='relu')(x)
    # x = Dropout(0.05)(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.05)(x)
    # x = Dense(512, activation='relu')(x)
    x = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=base_model.input, outputs=x)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


def load_dataset(path):
    images = []
    for file in os.listdir(path):
        if file.endswith(".jpg") or file.endswith(".png") or file.endswith(".jpeg"):
            # Read image
            image = cv2.imread(os.path.join(path, file))
            # Resize image
            resized = cv2.resize(image, (WIDTH, HEIGHT))
            # Scale down to 0-1
            image_scaled = resized / 255.0

            images.append(image_scaled)
    images = np.array(images)
    images = images.reshape(images.shape[0], HEIGHT, WIDTH, 3)
    return images


def load_labels(path):
    classes = []
    for file in os.listdir(path):
        if file.endswith(".json"):
            with open(os.path.join(path, file), 'r') as f:
                obj = json.load(f)
            classes.append([obj['class']])
    classes = np.array(classes)
    np.transpose(classes)
    classes = classes.reshape(classes.shape[0])
    return classes[::-1]


def runVideo():
    facedetector = tf.keras.models.load_model(FACE_DETECTOR)
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        _, frame = cap.read()
        frame = frame[50:500, 50:500, :]

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized = tf.image.resize(rgb, (120, 120))

        yhat = facedetector.predict(np.expand_dims(resized / 255, 0))

        if yhat[0] > 0.5:
            # Controls the text rendered
            cv2.putText(frame, "Face", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        cv2.imshow('EyeTrack', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()


if __name__ == '__main__':
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
        # tf.config.experimental.set_virtual_device_configuration(gpu, [
        #     tf.config.experimental.VirtualDeviceConfiguration(memory_limit=3072)])

    # IMAGES
    # x_humans = load_dataset('./data-gpt/X/Humans')
    # x_house = load_dataset('./data-gpt/X/house_data')
    # x_street = load_dataset('./data-gpt/X/street_data')
    # print(len(x_humans))
    # print(len(x_street))
    # print(len(x_house))
    # # LABELS
    # y_humans = load_labels('./data-gpt/Y/human_labels')
    # y_house = load_labels('./data-gpt/Y/house_labels')
    # y_street = load_labels('./data-gpt/Y/street_labels')
    # print(len(y_street))
    # print(len(y_house))
    # print(len(y_humans))

    # X = np.concatenate((x_humans, x_house, x_street))
    # Y = np.concatenate((y_humans, y_house, y_street))

    train_images = load_dataset(TRAIN_IMAGES)
    test_images = load_dataset(TEST_IMAGES)
    # val_images = load_dataset(VAL_IMAGES)

    train_labels = load_labels(TRAIN_LABELS)
    test_labels = load_labels(TEST_LABELS)
    # val_labels = load_labels(VAL_LABELS)

    # X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    # tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir='log')
    # facedetector = build_model()
    # hist = facedetector.fit(x=train_images, y=train_labels, epochs=20, batch_size=16)
    #
    # facedetector.save('facedetector_v4.h5')
    runVideo()


