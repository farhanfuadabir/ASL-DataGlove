import os
import cv2
import shutil
import tarfile
import requests
import argparse
import matplotlib
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, accuracy_score

matplotlib.use('Agg')

TEST_USER      = '001'
parser = argparse.ArgumentParser()
parser.add_argument("-t", "--test_user", type=int, choices=[1, 2, 3, 4, 5, 6, 7],
                    help="Test user to be used for evaluation")
args = parser.parse_args()
TEST_USER = '{:0>3d}'.format(args.test_user)

os.system('figlet \'Talk to da Hand\'')

DATASET_ID     = '1cAJdvAZDolurN3KCZcYz_YJSMV-aIzWT'

# -------------BASE DIR (MODIFY THIS TO YOUR NEED) ------------ #
BASE_DIR       = '.'
IMG_DIR        = 'BW-Spatial-Path-Images/'
LOG_DIR        = 'Logs/'

DATA_DIR       = 'Sensor-Data/'
BW_IMG_DIR     = 'BW-Spatial-Path-Images/'
RGB_IMG_DIR    = 'RGB-Spatial-Path-Images/'
IMG_SIZE       = (3, 3) # INCHES

USERS          = ['001', '002', '003', '004', '005', '006', '007']
# ------------------------------- Only Dynalic Gestures ------------------------------ #
GESTURES       = ['j', 'z', 'bad', 'deaf', 'fine', 'good', 'goodbye', 'hello', 'hungry',
                'me', 'no', 'please', 'sorry', 'thankyou', 'yes', 'you']

PLANES         = ['XY', 'YZ', 'ZX']

DT             = 0.01
LINEWIDTH      = 7

BATCH_SIZE     = 32
IMG_LEN        = 160
IMG_SIZE       = (IMG_LEN, IMG_LEN)

# ------------- FOR THE GREATER GOOD :) ------------- #
TRAIN_LEN      = 960
TEST_LEN       = 160

EPOCHS         = 7
LEARNING_RATE  = 0.001

IMG_SHAPE = IMG_SIZE + (3,)

BASE_MODEL = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE, include_top=False, weights='imagenet')
BASE_MODEL.trainable = False



#--------------------- Download util for Google Drive ------------------- #

def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)
        
    save_response_content(response, destination)    

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value
        
    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:
                f.write(chunk)

def download_data(fid, destination):
    print('cleaning already existing files ... ', end='')
    try:
        shutil.rmtree(destination)
        print('√')
    except:
        print('✕')
        
    print('creating data directory ... ', end='')
    os.mkdir(destination)
    print('√')
    
    print('downloading dataset from the repository ... ', end='')
    filename = os.path.join(destination, 'dataset.tar.xz')
    try:
        download_file_from_google_drive(fid, filename)
        print('√')
    except:
        print('✕')
        
    print('extracting the dataset ... ', end='')
    try:
        tar = tarfile.open(filename)
        tar.extractall(destination)
        tar.close()
        print('√')
    except:
        print('✕')   


# ------------- Spatial Path Image Generation ----------- #

def clean_dir(path):
    print('cleaning already existing files ... ', end='')
    try:
        shutil.rmtree(path)
        print('√')
    except:
        print('✕')
    
    print('creating ' + path + ' directory ... ', end='')
    os.mkdir(path)
    print('√')
    
# ----------- Spatial Path Vector Calculation ----------- #

def get_displacement(acc):
    v = np.zeros(acc.shape)
    d = np.zeros(acc.shape)
    for i in range(acc.shape[0] - 1):
        v[i + 1] = v[i] + acc[i] * DT
        d[i + 1] = v[i] * DT + 0.5 * acc[i] * DT * DT
        
    return d

def write_image(x, y, path):
    _, ax = plt.subplots(frameon=True, figsize=(3, 3))
    ax.axis('off')
    plt.plot(x, y, '-k', linewidth=LINEWIDTH)
    plt.savefig(path)
    plt.close()

def generate_bw_images():
    count = 0
    image_dir = os.path.join(BASE_DIR, BW_IMG_DIR)
    clean_dir(image_dir)
    
    for plane in PLANES:
        print('processing spatial path images for ' + plane + ' plane ... ', end='')
        plane_dir = os.path.join(image_dir, plane)
        os.mkdir(plane_dir)
        
        for gesture in GESTURES:
            os.mkdir(os.path.join(plane_dir, gesture))
    
            for user in USERS:
                user_dir = os.path.join(BASE_DIR, DATA_DIR, user)
                gesture_dir = os.path.join(user_dir, gesture + '.csv')
                
                accx = pd.read_csv(gesture_dir)['ACCx_world'].to_numpy()
                accy = pd.read_csv(gesture_dir)['ACCy_world'].to_numpy()
                accz = pd.read_csv(gesture_dir)['ACCz_world'].to_numpy()

                x = get_displacement(accx).reshape(-1, 150)
                y = get_displacement(accy).reshape(-1, 150)
                z = get_displacement(accz).reshape(-1, 150)

                for i in range(x.shape[0]):
                    image_name = 'u' + user + '_g' + '{:0>2d}'.format(GESTURES.index(gesture)) +                                  '_s' + '{:0>7d}'.format(count) + '_p' + plane + '.jpg'
                    path = os.path.join(BASE_DIR, BW_IMG_DIR, plane, gesture, image_name)
                    
                    if plane == 'XY':
                        write_image(x[i, :], y[i, :], path)
                    elif plane == 'YZ':
                        write_image(y[i, :], z[i, :], path)
                    else:
                        write_image(z[i, :], x[i, :], path)

                    count = count + 1
            
        print('√')

def load_data(plane):
    X_train = np.zeros((TRAIN_LEN, IMG_LEN, IMG_LEN, 3))
    X_test = np.zeros((TEST_LEN, IMG_LEN, IMG_LEN, 3))
    y_train = np.zeros((TRAIN_LEN, 1))
    y_test = np.zeros((TEST_LEN, 1))
    
    train_count = 0
    test_count = 0
        
    for gesture in GESTURES:
        print('loading data for ' + gesture + ' gesture on the ' + plane + ' plane ... ', end='')
        path = os.path.join(BASE_DIR, IMG_DIR, plane, gesture)
        for filename in os.listdir(path):
            img = cv2.imread(os.path.join(path, filename))
            resized = cv2.resize(img, IMG_SIZE)
            if filename[1:4] != TEST_USER:
                X_train[train_count, :] = resized
                y_train[train_count, 0] = GESTURES.index(gesture)
                train_count = train_count + 1
            else:
                X_test[test_count, :] = resized
                y_test[test_count, 0] = GESTURES.index(gesture)
                test_count = test_count + 1
                
        print('√')
        
    return X_train, X_test, y_train, y_test

def get_model():
    preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input
    rescale = tf.keras.layers.experimental.preprocessing.Rescaling(1./127.5, offset= -1)
    global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
    prediction_layer = tf.keras.layers.Dense(len(GESTURES))
    inputs = tf.keras.Input(shape=IMG_SHAPE)
    x = preprocess_input(inputs)
    x = BASE_MODEL(x, training=False)
    x = global_average_layer(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = prediction_layer(x)
    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=LEARNING_RATE),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
    return model


def main():
    # ------- Comment This if already downloaded -------- #
    destination = os.path.join(BASE_DIR, DATA_DIR)
    download_data(DATASET_ID, destination)

    generate_bw_images()

    X_train_xy, X_test_xy, y_train_xy, y_test_xy = load_data('XY')
    X_train_yz, X_test_yz, y_train_yz, y_test_yz = load_data('YZ')
    X_train_zx, X_test_zx, y_train_zx, y_test_zx = load_data('ZX')

    print('Processing XY plane ... ')
    print('======================================================')
    model_xy = get_model()
    _ = model_xy.fit(X_train_xy, y_train_xy, epochs=EPOCHS)   

    prob_xy = tf.keras.Sequential([model_xy, tf.keras.layers.Softmax()])
    y_pred_xy = prob_xy.predict(X_test_xy)
    y_pred = np.argmax(y_pred_xy, axis=1)
    print(classification_report(y_test_xy.ravel(), y_pred))

    print('Processing YZ plane ... ')
    print('======================================================')
    model_yz = get_model()
    _ = model_yz.fit(X_train_yz, y_train_yz, epochs=EPOCHS)

    prob_yz = tf.keras.Sequential([model_yz, tf.keras.layers.Softmax()])
    y_pred_yz = prob_yz.predict(X_test_yz)
    y_pred = np.argmax(y_pred_yz, axis=1)
    print(classification_report(y_test_yz.ravel(), y_pred))

    print('Processing ZX plane ... ')
    print('======================================================')
    model_zx = get_model()
    _ = model_zx.fit(X_train_zx, y_train_zx, epochs=EPOCHS)

    prob_zx = tf.keras.Sequential([model_zx, tf.keras.layers.Softmax()])
    y_pred_zx = prob_zx.predict(X_test_zx)
    y_pred = np.argmax(y_pred_zx, axis=1)
    print(classification_report(y_test_zx.ravel(), y_pred))

    print('Combining All the Planes ... ')
    print('======================================================')
    y_total = y_pred_xy * y_pred_yz * y_pred_zx
    y_pred = np.argmax(y_total, axis=1)
    report = classification_report(y_test_xy.ravel(), y_pred, zero_division=0)
    print(report)

    config = '\n\nTEST_USER ' + TEST_USER + '\n'
    underline = '=====================================\n'
    log_dir = os.path.join(BASE_DIR, LOG_DIR)
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    f = open(os.path.join(log_dir, 'logs.txt'), 'a')
    f.write(config)
    f.write(underline)
    f.write(report)
    f.close()


if __name__ == '__main__':
    main()
