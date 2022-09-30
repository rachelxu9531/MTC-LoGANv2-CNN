from keras.models import Sequential
import tensorflow as tf
#tf.enable_eager_execution()

from keras.layers.core import Dense, Activation, Dropout, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import Adam, RMSprop,  SGD
from keras.utils import np_utils
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn import metrics
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
import random
from numpy import *
from PIL import Image
import theano
import pandas as pd
#%matplotlib inline
import pandas as pd
import scipy.ndimage
import os.path
import sys
import time
import scipy.io as sio
import random
import copy
from scipy import stats
from scipy.interpolate import griddata
from matplotlib import pyplot as plt, colors
import seaborn as sns
from PIL import Image
import tkinter as tk
from tkinter import simpledialog
import matplotlib.ticker as ticker

path_train = "C:\Conditional-StyleGAN\CNN\thebe3d\Training\Trial_3"
path_train = "C:\Conditional-StyleGAN\CNN\thebe3d\Testing\Trial_3"

#Select file and auto load the name of files
from tkinter import Tk, Button, filedialog, Label

root = Tk()
y = ""
root.title("test")

def openfiledialog():
    global y
    y = filedialog.askopenfilename(initialdir = "/",title = "Select file",filetypes = (("text file","*.txt"),("all files","*.*")))

def check_path():
    global y
    print(y)

file_button = Button(root, text='Select File', command=openfiledialog)
file_button.pack()

exit_button = Button(root, text='Exit Program', command=root.destroy)
exit_button.pack()
w = Label(root, text="\nStep 1: Make sure the URL's in file are properly formatted, one URL per line\n\nStep 2: Select file\n\nStep 3: click run")
w.pack()

Button(root, text="Print current saved path", command = check_path).pack()

root.mainloop()
file_name=y.split("/thebe3d/")[1].split("/")[0]
print(file_name)

#fig_title='Seabed variance minus ' + file_name.split("-")[1].split("/")[0]

#file_name = 'C:/Conditional-StyleGAN/CNN/Testing/Extract value^Thebe3d_pstm_Time [Var] [Realized] 1; -50'
#np.loadtxt is slow, we use pandas here
#df = pd.read_csv(y, delimiter=' ')
df = pd.read_csv(file_name, delimiter=' ')
print(df)

X = df.values[:, 0]
Y = df.values[:, 1]
Z = df.values[:, 2]

X = df.values[:, 0]
Y = df.values[:, 1]
Z = df.values[:, 2]

#Some statistics of the datasets
min_X=np.min(X);max_X=np.max(X)
min_Y=np.min(Y);max_Y=np.max(Y)
min_Z=np.min(Z);max_Z=np.max(Z)
print(min_X, max_X, min_Y, max_Y, min_Z, max_Z)

#The original seismic boundary is recorded at a angle of 7.833. This value however does not offset the figure
#Thus, the calcultion is based on the half of the angle that between min_X and min_Y
ind_min_X=np.where(X==min_X)
ind_min_Y=np.where(Y==min_Y)
print(ind_min_X, ind_min_Y)

#Calculation of theta
theta=-0.5*np.arctan((Y[ind_min_Y]-Y[ind_min_X])/(X[ind_min_Y]-X[ind_min_X]))
print(theta) #This theta is in rads

#Rotate the axis
Xnew = X*np.cos(theta)-Y*np.sin(theta)
Ynew =X*np.sin(theta)+Y*np.cos(theta)
#Ynew = -Ynew

print(np.min(Xnew), np.max(Xnew))
print(np.min(Ynew), np.max(Ynew))

print(Xnew[1:10]-Xnew[:9])        # This might help to check if the points are parallel to X and Y
print(Ynew[1:10]-Ynew[:9])


# Combine two colormaps - similar to Petrel
my_cmap1 = plt.cm.Greys(np.linspace(0, 1, 128))
my_cmap2 = plt.cm.hot(np.linspace(0, 1, 128))

# combine them and build a new colormap
jointcolors = np.vstack((my_cmap1[:-5], my_cmap2[:-15]))
my_cmap = colors.LinearSegmentedColormap.from_list('my_colormap', jointcolors)

N = Xnew.shape[0]

# For faster plotting only
# skip and skip2 should be 1 (no skipping) in the final tests!
skip = 1

#plt.style.use('seaborn-darkgrid')
fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(12,10), dpi=300)
fig.subplots_adjust(wspace=0.33, hspace=0.5)

sc = ax.scatter(Xnew[::skip], Ynew[::skip], c=np.log10(Z[::skip]), marker='s', alpha=1, linewidths=0.0, edgecolors='k', cmap="Reds", s=3)
ax.set_title('Seabed variance minus 50', fontsize=11)
ax.set_xlabel('X')
ax.set_ylabel('Y')
fig.colorbar(sc, ax=ax, aspect=50, pad=0.03)
plt.savefig('Overview_play.png', bbox_inches='tight', dpi=600)

#auto assign input variables
#skip should be 1 in the actual test. For quick plotting, it can be setted to any number
from tkinter import Tk, Label, IntVar, Entry, Button
root = Tk()
title = Label(root, text="Input dialog")
title.pack()

#x_step
Input1_label = Label(root, text="x_step:")
Input1_label.pack()
Input1 = IntVar()
Input1_entry = Entry(textvariable=Input1)
Input1_entry.pack()

#y_step
Input2_label = Label(root, text="y_step:")
Input2_label.pack()
Input2 = IntVar()
Input2_entry = Entry(textvariable=Input2)
Input2_entry.pack()

#skip
Input3_label = Label(root, text="skip:")
Input3_label.pack()
Input3 = IntVar()
Input3_entry = Entry(textvariable=Input3)
Input3_entry.pack()

def print_fun():
    print(Input1.get(),Input2.get(),Input3.get())

go_button = Button(root, text='Go', command=print_fun)
go_button.pack()

exit_button = Button(root, text='Exit Program', command=root.destroy)
exit_button.pack()
#
root.mainloop()

x_step = int(Input1.get()) #10    # in meters
y_step = int(Input2.get()) #10
skip2 = int(Input3.get())     # ideally this should be one

start = time.time()
grid_x, grid_y = np.mgrid[np.min(Xnew):np.max(Xnew):x_step, np.min(Ynew):np.max(Ynew):y_step]
interpolated = griddata((Xnew[::skip2], Ynew[::skip2]), Z[::skip2],
                   (grid_x, grid_y), method='linear') # nearest, linear, cubic
# Fliplr it and transpose for future plotting
interpolated = np.fliplr(interpolated).T
datamin = 0.0001                              # this number comes from the data #previous version is defined as 0.0002
datamax = 1

interpolated[np.isnan(interpolated)] = datamin
interpolated[interpolated<datamin] = datamin
interpolated[interpolated>datamax] = datamax

print(interpolated.shape)
end = time.time()
print('Time:', end - start, 'sec')

print(interpolated[1].shape)

#set tile size
#this number is normally the power of 2
root = Tk()
mv = Label(root, text="Tile dialog")
mv.pack()

#window size
mv1_label = Label(root, text="Tile size:")
mv1_label.pack()
mv1 = IntVar()
mv1_entry = Entry(textvariable=mv1)
mv1_entry.pack()

def mv_print_fun():
    print(mv1.get())

mv_go_button = Button(root, text='Go', command=mv_print_fun)
mv_go_button.pack()

mv_exit_button = Button(root, text='Exit Program', command=root.destroy)
mv_exit_button.pack()

root.mainloop()

# 1 - Featureless low-variance
# 2 - Sinuous high-variance
# 3 - Linear moderate-variance
# 4 - Circular low-variance

# Moving window
def make_windows_from_section(data1, Nx, Ny, sizex, sizey):
    windows = np.zeros((Nx*Ny, sizey, sizex))
    for i in range(Ny):
        for j in range(Nx):
            windows[i*Nx+j,:,:] = data1[i*sizey:(i+1)*sizey, j*sizex:(j+1)*sizex]
    return windows

window_size = int(mv1.get()) #128
Nx, Ny = np.shape(interpolated)[1]//window_size, np.shape(interpolated)[0]//window_size
windows = make_windows_from_section(interpolated, Nx, Ny, window_size, window_size)

Nwindows = windows.shape[0]
print(Nwindows, 'windows of', windows.shape[1:], 'generated')

# use global max and min for them
min1 = np.min(windows)
max1 = np.max(windows)
print('Global min/max:', min1, max1)

model = tf.keras.models.load_model('RX_model.hdf5') #loading the model
model.summary

file_path='C:\Conditional-StyleGAN\CNN\img'

from matplotlib import cm
import PIL.Image
import cv2
import pickle
import os
from PIL import Image

dataset = []
for i in range(Nwindows):
    filename = file_path + str(i) + '.png'

    # local min/max
    # values8 = (((windows[i] - windows[i].min()) / (windows[i].max() - windows[i].min())) * 255.99).astype(np.uint8)

    # global min/max
    # values8 = (((windows[i] - min1) / (max1 - min1)) * 255.99).astype(np.uint8)

    # USE THIS - global min/max and log 10
    values8 = (((np.log10(windows[i]) - np.log10(min1)) / (np.log10(max1) - np.log10(min1))) * 255.99).astype(np.uint8)
    # print(symbol, 'min/max:', np.min(values8), np.max(values8))
    # img = Image.fromarray(values8)
    img = Image.fromarray(np.uint8(cm.Reds(values8) * 255.99))  # plasma
    # img = Image.fromarray(values8)
    # .transpose(Image.FLIP_LEFT_RIGHT)
    # img.save(filename, cmap="Reds")#, origin='lower')

    img.save(filename, cmap="Reds")  # , origin='lower')
    img = np.asarray(img)
    if len(img.shape) > 2 and img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    img = Image.fromarray(img)
    img = img.convert("RGB")
    img.save(file_name, 'png')
    img_array = cv2.imread(file_name)
    dataset.append([img_array])

    X_test = []
    for features in dataset:
        X_test.append(features)
        # y_test.append(label)
    X_test = np.array(X_test).reshape(-1, 128, 128, 3)
    X_test = X_test.astype('float32')
    X_test /= 255

    y_pred = model.predict(X_test)

    from sklearn.metrics import classification_report

    y_pred = np.argmax(y_pred, axis=1)
    print(y_pred.shape)

    my_dict = dict()
    for index in range(y_pred.shape[0]):
        my_dict[index] = windows[index]
    print(my_dict)

    y_pred_re = y_pred.reshape(Ny, Nx)
    print(y_pred_re)

    # define color map
    color_map = {0: np.array([255, 0, 0]),  # red
                 1: np.array([255, 255, 0]),  # green
                 2: np.array([255, 0, 255]),
                 3: np.array([0, 255, 0]),
                 4: np.array([0, 255, 255]),
                 5: np.array([255, 255, 30]),
                 6: np.array([30, 70, 40]),
                 7: np.array([0, 90, 0]),
                 8: np.array([90, 0, 30])}
    print(color_map)
    # make a 3d numpy array that has a color channel dimension
    data_3d = np.ndarray(shape=(Ny, Nx, 3), dtype=int)
    print(data_3d.shape)

    for i in range(0, Ny):
        for j in range(0, Nx):
            data_3d[i][j] = color_map[y_pred_re[i][j]]
    # display the plot
    fig, ax = plt.subplots(1, 1)
    ax.imshow(data_3d)
    import matplotlib.patches as patches


    # plot the trace of each window for illustration purpose
    def plot_windows(windows, Nx, Ny):
        fig = plt.figure(figsize=(0.8 * Nx, 0.8 * Ny), dpi=100)
        for i in range(0, Ny):
            for j in range(0, Nx):
                ax = fig.add_subplot(Ny, Nx, i * Nx + j + 1)

                ax.imshow(np.log10(windows[i * Nx + j, :, :]), cmap='Greys', aspect='auto', vmin=np.log10(min1),
                          vmax=np.log10(max1))
                square = patches.Rectangle((0, 0), 128, 128, linewidth=0.25, edgecolor='r',
                                           facecolor=data_3d[i][j] / 255, alpha=0.5)
                ax.add_patch(square)

                # add color label for that window
                # draw a transparent square of color      "colors[CLASS[i]]"
                ax.set_axis_off()

        fig.tight_layout()
        fig.subplots_adjust(wspace=0.025, hspace=0.025)
        plt.show()


    plot_windows(windows, Nx, Ny)

    CATEGORIES = ['Blocks', 'Extensional ridges', 'Grooves and striations', 'Individual flow', 'MTC material',
                  'Polygonally faults',
                  'Scarps', 'Slump folds', 'Undisturbed']
    IMG_SIZE = 128
    nb_classes = 9


    def creatData(path, category):
        dataset = []
        for i in category:
            path_r = os.path.join(path, i)
            class_num = category.index(i)
            for img in os.listdir(path_r):
                img_array = cv2.imread(os.path.join(path_r, img))
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                dataset.append([new_array, class_num])
        return (dataset)


    training = creatData(path_train, CATEGORIES)
    testing = creatData(path_test, CATEGORIES)

    random.shuffle(training)
    random.shuffle(testing)

    X_train = []
    y_train = []
    for features, label in training:
        X_train.append(features)
        y_train.append(label)
    X_train = np.array(X_train).reshape(-1, IMG_SIZE, IMG_SIZE, 3)
    X_train = X_train.astype('float32')
    X_train /= 255
    y_train = np.array(y_train)
    print(np.shape(X_train))
    print(y_train)

    X_test = []
    y_test = []
    for features, label in testing:
        X_test.append(features)
        y_test.append(label)
    X_test = np.array(X_test).reshape(-1, IMG_SIZE, IMG_SIZE, 3)
    X_test = X_test.astype('float32')
    X_test /= 255
    y_test = np.array(y_test)
    print(np.shape(X_test))
    print(y_test)

    batch_size = 16
    nb_epochs = 20
    img_rows, img_columns = 128, 128
    img_channel = 3
    nb_filters = 32
    nb_pool = 2
    nb_conv = 3
    model = tf.keras

    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation=tf.nn.relu,
                               input_shape=(128, 128, 3)),
        tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation=tf.nn.relu,
                               input_shape=(128, 128, 3)),
        tf.keras.layers.MaxPooling2D((2, 2), strides=2),
        tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation=tf.nn.relu,
                               input_shape=(128, 128, 3)),
        tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation=tf.nn.relu,
                               input_shape=(128, 128, 3)),
        tf.keras.layers.MaxPooling2D((2, 2), strides=2),
        tf.keras.layers.Conv2D(128, (3, 3), padding='same', activation=tf.nn.relu),
        tf.keras.layers.Conv2D(128, (3, 3), padding='same', activation=tf.nn.relu),
        tf.keras.layers.MaxPooling2D((2, 2), strides=2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation=tf.nn.relu),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(128, activation=tf.nn.relu),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(128, activation=tf.nn.relu),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(9, activation=tf.nn.softmax)
    ])
    model.summary()

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(X_train, y_train, batch_size=batch_size, epochs=nb_epochs, verbose=1,
                        validation_data=(X_test, y_test))

    plt.style.use("ggplot")
    fig0 = plt.figure()
    plt.plot(np.arange(0, nb_epochs), history.history["accuracy"], label="train_acc")
    plt.plot(np.arange(0, nb_epochs), history.history["val_accuracy"], label="val_acc")
    plt.plot(np.arange(0, nb_epochs), history.history["loss"], label="train_loss")
    plt.plot(np.arange(0, nb_epochs), history.history["val_loss"], label="val_loss")
    plt.title("Training and Validation of Loss and Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="upper right")
    plt.grid(False)

    plt.tight_layout()
    plt.show()
    fig0.savefig('training history.png', dpi=300)

    tf.keras.models.save_model(model, 'RX_model.hdf5')  # saving the model

    model = tf.keras.models.load_model('RX_model.hdf5')  # loading the model


    # Evaluating the model on test and train data to finding the loss and accuracy
    results = model.evaluate(X_test, y_test)
    # Accuracy and loss for test data
    print("Loss of the model  while using  test datas ", results[0])
    print("Accuracy of the model while using  test datas t", results[1] * 100, "%")
    results = model.evaluate(X_train, y_train)
    # Accuracy and loss for Train data
    print("Loss of the model  while using  train datas  ", results[0])
    print("Accuracy of the model while using  train datas ", results[1] * 100, "%")

    y_pred = model.predict(X_test)  # predicting the label values

    from sklearn.metrics import classification_report

    y_pred = np.argmax(y_pred, axis=1)
    print(classification_report(y_test, y_pred, target_names=CATEGORIES))  # printing classification metrics

    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_test, y_pred)

    import seaborn as sns
    import matplotlib.pyplot as plt

    fig1 = plt.figure(figsize=(12, 9))

    ax = plt.subplot()
    sns.set(font_scale=1.2)
    sns.heatmap(cm, annot=True, fmt='g', ax=ax);  # annot=True to annotate cells, ftm='g' to disable scientific notation

    # labels, title and ticks
    ax.set_xlabel('Predicted');
    ax.set_ylabel('True');
    ax.set_title('Confusion Matrix');
    ax.tick_params(labelsize=14)
    ax.xaxis.set_ticklabels(CATEGORIES, ha="center", rotation=90);
    ax.yaxis.set_ticklabels(CATEGORIES, va="center", rotation=0);

    plt.tight_layout()
    plt.show()
    fig1.savefig('cm.png', dpi=300)