import math
import tkinter
import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from tkinter import Tk, ttk, filedialog, Text
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score

tk = Tk()
tk.geometry("450x250")
tk.resizable(False, False)
tk.title("Image Processing Project")

pixels_array = []
histogram = []
labels = []
normalized_histogram = []


def get_images():
    path = glob.glob("./flowers_dataset/*")
    for i in range(len(path)):
        sub_path = glob.glob(path[i] + "/*")
        for j in range(len(sub_path)):
            if feature_extraction.get() == "HOG":
                hog(path=sub_path[j])
            if feature_extraction.get() == "LBP":
                lbp(path=sub_path[j])

    for dandelion in range(401):
        labels.append(0)
    for rose in range(324):
        labels.append(1)
    for sunflower in range(454):
        labels.append(2)

    classification()


def demo_dataset_process():
    path = glob.glob("./flowers_dataset/*")
    for i in range(len(path)):
        sub_path = glob.glob(path[i] + "/*")
        for j in range(5):
            if feature_extraction.get() == "HOG":
                hog(path=sub_path[j])
            if feature_extraction.get() == "LBP":
                lbp(path=sub_path[j])

    for dandelion in range(5):
        labels.append(0)
    for rose in range(5):
        labels.append(1)
    for sunflower in range(5):
        labels.append(2)


# classification()


def test_image():
    filetypes = (('Image Files', '*.png *.jpg *.jpeg'), ('All files', '*.*'))
    file_path = filedialog.askopenfilename(initialdir='./Desktop', filetypes=filetypes)
    labels.append([0])
    if feature_extraction.get() == "HOG":
        hog(file_path)
    if feature_extraction.get() == "LBP":
        lbp(file_path)

    # classification()
    labels.clear()
    histogram.clear()


def lbp(path):
    lbp_image = Image.open("lbp_image.png").convert("L").resize((64, 128))
    file = Image.open(path).convert("L").resize((64, 128))

    lbp_array = []
    idx_array = [[1, -1], [1, 0], [1, 1], [0, 1], [-1, 1], [-1, 0], [-1, -1], [0, -1]]
    for height in range(1, file.height - 1):
        for width in range(1, file.width - 1):
            for idx in range(len(idx_array)):
                if file.getpixel((width, height)) >= file.getpixel(
                        (width + idx_array[idx][0], height + idx_array[idx][1])):
                    lbp_array.append(1)
                else:
                    lbp_array.append(0)

            new_pixel = 0
            new_pixel_array = []
            for index in range(8):
                if lbp_array[index] == 1:
                    new_pixel += 2 ** index
                    pixels_array.append(new_pixel)
                    new_pixel_array.append(new_pixel)
            lbp_image.putpixel((width, height), new_pixel)
            lbp_array.clear()

    lbp_hist = []
    for i in range(256):
        lbp_hist.append(pixels_array.count(i))

    # lbp_image.resize((500, 500)).show()


def lbp_reset():
    lbp_image = Image.open("lbp_image.png").resize((64, 128))
    for height in range(lbp_image.height):
        for width in range(lbp_image.width):
            lbp_image.putpixel((width, height), (255, 255, 255, 255))
    lbp_image.save("./lbp_image.png")


def hog(path):
    im = cv2.imread(path)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    im = cv2.resize(im, (64, 128))

    gx = cv2.Sobel(im, cv2.CV_32F, 1, 0, ksize=1)
    gy = cv2.Sobel(im, cv2.CV_32F, 0, 1, ksize=1)
    mag, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)

    gradient_array = [0, 0, 0, 0, 0, 0, 0, 0, 0]

    row = 8
    column = 8
    for i in range(16):
        for j in range(8):
            for y in range(column - 8, column):
                for x in range(row - 8, row):
                    sub_mag = (mag[y][x]) / 20
                    sub_angle = angle[y][x] / 2
                    for value in range(0, 161, 20):
                        if value < sub_angle < (value + 20):
                            left_mag = sub_mag * (sub_angle - value)
                            right_mag = sub_mag * ((value + 20) - sub_angle)

                            gradient_array[int(value / 20)] += round(left_mag)

                            if int((value + 20) / 20) == 9:
                                gradient_array[0] += round(right_mag)
                            else:
                                gradient_array[int((value + 20) / 20)] += round(right_mag)

            histogram.append(gradient_array)
            gradient_array = [0, 0, 0, 0, 0, 0, 0, 0, 0]
            row = row + 8
        column = column + 8
        row = 8

    new_value_hist = []
    index = 0
    for a in range(15):
        for b in range(7):
            print(histogram[index])
            print(histogram[index + 1])
            print(histogram[index + 8])
            print(histogram[index + 9])
            n_hist = histogram[index] + histogram[index + 1] + histogram[index + 8] + histogram[index + 9]
            print(n_hist)
            n_hist_sum = 0
            for n in range(36):
                n_hist_sum += (n_hist[n] * n_hist[n])
            n_sqrt_value = math.sqrt(n_hist_sum)
            for n_sqrt in range(36):
                new_value_hist.append(n_hist[n_sqrt] / n_sqrt_value)
            index += 1
            normalized_histogram.append(new_value_hist)
            new_value_hist.clear()


def classification():
    x = normalized_histogram
    y = labels
    x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                        test_size=int(test_percent.get()) / 100,
                                                        train_size=int(train_percent.get()) / 100)
    x_test = np.asarray(x_test)

    if classification_combobox.get() == "KNN":
        knn = KNeighborsClassifier(n_neighbors=3)
        knn.fit(x_train, y_train)
        knn_predict_ = knn.predict(x_test)

        cm_text.insert(tkinter.END, "Confusion Matrix: \n" + str(confusion_matrix(y_test, knn_predict_)) + "\n")
        acc_text.insert(tkinter.END, "Accuracy: " + str(accuracy_score(y_test, knn_predict_)) + "\n")

        print("Confusion Matrix: \n", confusion_matrix(y_test, knn_predict_))
        print("Accuracy: ", accuracy_score(y_test, knn_predict_))

        plt.figure()
        plt.scatter(x_test[:, 0], x_test[:, 1], c=y_test, alpha=0.75)
        plt.show()

    if classification_combobox.get() == "SVM":
        svm = LinearSVC(C=100, random_state=42)
        svm.fit(x_train, y_train)
        svm_predict_ = svm.predict(x_test)

        cm_text.insert(tkinter.END, "Confusion Matrix: \n" + str(confusion_matrix(y_test, svm_predict_)) + "\n")
        acc_text.insert(tkinter.END, "Accuracy: \n" + str(accuracy_score(y_test, svm_predict_)) + "\n")

        print("Confusion Matrix: \n", confusion_matrix(y_test, svm_predict_))
        print("Accuracy: ", accuracy_score(y_test, svm_predict_))

        plt.figure()
        plt.scatter(x_test[:, 0], x_test[:, 1], c=y_test, alpha=0.75)
        plt.show()


feature_extraction_label = ttk.Label(master=tk, text='Feature Extraction')
feature_extraction_label.place(x=10, y=30)

feature_extraction = ttk.Combobox(master=tk, width=15)
feature_extraction['values'] = ('HOG', 'LBP')
feature_extraction.current(0)
feature_extraction.place(x=10, y=50)

classification_label = ttk.Label(master=tk, text='Classification')
classification_label.place(x=10, y=80)

classification_combobox = ttk.Combobox(master=tk, width=15)
classification_combobox['values'] = ('KNN', 'SVM')
classification_combobox.current(0)
classification_combobox.place(x=10, y=100)

train_percent_label = ttk.Label(master=tk, text='Train Percent')
train_percent_label.place(x=150, y=30)

train_percent = ttk.Entry(master=tk, width=18)
train_percent.insert(0, str(80))
train_percent.place(x=150, y=50)

test_percent_label = ttk.Label(master=tk, text='Test Percent')
test_percent_label.place(x=150, y=80)

test_percent = ttk.Entry(master=tk, width=18)
test_percent.insert(0, str(20))
test_percent.place(x=150, y=100)

cm_text = Text(width=17, height=5, master=tk)
cm_text.place(x=10, y=130)

acc_text = Text(width=17, height=5, master=tk)
acc_text.place(x=150, y=130)

start_process_btn = ttk.Button(text="Start Process", width=17, command=get_images)
start_process_btn.place(x=300, y=50)

test_image_btn = ttk.Button(text="Load Test Image", width=17, command=test_image)
test_image_btn.place(x=300, y=100)

demo_image_btn = ttk.Button(text="Small Dataset", width=17, command=demo_dataset_process)
demo_image_btn.place(x=300, y=150)

tk.mainloop()
