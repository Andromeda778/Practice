import numpy as np
import tkinter as tk
from tkinter import *
from PIL import Image, ImageTk
import cv2
import tensorflow as tf
from keras import backend as K

root = tk.Tk()
root.geometry("640x560")
root.title("ConvoMed")
root.iconbitmap("caduc.ico")


upp_frame = Frame(root, bd = 15)
upp_frame.pack()
mid_frame = Frame(root, bd =15)
mid_frame.pack()
bot_frame = Frame(root, bd = 15)
bot_frame.pack()
dialog_frame = Frame(root, bd = 15)
dialog_frame.pack()


def model_open(initialdir='/'):
    from tkinter.filedialog import askopenfile, askopenfilename
    file_path  = askopenfilename(initialdir=initialdir,
                                filetypes = [ ('Model', '*.h5') ]  )
    okno.set("Выбор модели")
    cnn_var.set(file_path)

    return file_path

def model_load():
    from tensorflow.keras.models import load_model, Model
    okno.set("Загрузка модели...")
    global model_cnn
    weight_shortcut = cnn_entry.get()
    model_cnn = load_model(weight_shortcut)
    model_cnn.summary()

    okno.set("Модель загружена!")

    return
    


def open_pic(initialdir='/'):
    from tkinter.filedialog import askopenfile, askopenfilename
    file_path  = askopenfilename(initialdir=initialdir, filetypes = [ ('Image', '*.jpeg*' ) ]  )
    okno.set("Выбор изображения")
    pic_var.set(file_path)

    image = Image.open(file_path)
    image = image.resize((224,224))
    photo = ImageTk.PhotoImage(image)

    pic_label = Label(mid_frame, image=photo, padx=10, pady=10)
    pic_label.image = photo
    pic_label.grid(row=3, column=1)

    return file_path

def load_pic():
    okno.set("Загрузка изображения...")
    path = pic_entry.get()
    global pics

    pics = cv2.imread(path)
    pics = cv2.resize(pics,(224, 224))
    pics = np.reshape(pics,(1, 224, 224, 3))
    pics = np.array(pics) / 255
    print(pics.shape)
    okno.set("Изображение загружено!")

    return


def test_pic():

    model_cnn.compile(loss='categorical_crossentropy', metrics=['accuracy'],optimizer='adam')

    import time
    first = time.time()
    predictions = model_cnn.predict(pics)
    second = time.time()
    with open('logs.txt', 'a', encoding='utf-8') as file:
        result_text = "Наиболее вероятный вид: " +str(predictions)
        result_okno.set(result_text)
        print("Результат вычисления: " +str(predictions), file=file)
        okno.set("Готово!")
 

cnn_var = StringVar()
cnn_var.set("/")
cnn_entry = Entry(upp_frame, textvariable=cnn_var, width=45)
cnn_entry.grid(row=2, column=2)

cnn_load_btn = Button(upp_frame, text='Выбрать модель',  command =  lambda: model_open(cnn_entry.get()), bg="black", fg="white" )
cnn_load_btn.grid(row=2, column=1)


cnn_confirm_btn = Button(upp_frame, text='Загрузить модель',  command = model_load , bg="black", fg="white" )
cnn_confirm_btn.grid(row=2, column=4)

pic_var = StringVar()
pic_var.set("/")
pic_entry = Entry(upp_frame, textvariable=pic_var, width=45)
pic_entry.grid(row=7, column=2)

pic_load_btn = Button(upp_frame, text='Выбрать изображение',  command =  lambda: open_pic(pic_entry.get()), bg="black", fg="white" )
pic_load_btn.grid(row=7, column=1)

pic_confirm_btn = Button(upp_frame, text='Загрузить изображение',  command = load_pic , bg="black", fg="white" )
pic_confirm_btn.grid(row=7, column=4)


proverka_btn = Button(bot_frame, text='Проверить', command = test_pic , bg="black", fg="white" )
proverka_btn.pack()

result_okno = StringVar()
result_okno.set("Полученный результат")
label_test = Label(bot_frame,font=("Arial", 20), height=3, textvariable=result_okno, bg="white", fg="black").pack()

okno = StringVar()
okno.set("Добро пожаловать в ConvoMed!")

lframe1 = LabelFrame(dialog_frame)
lframe1.pack()

labletop = Label(lframe1,font=("Arial", 17), height=2, textvariable=okno, fg="red", bg="white")
labletop.pack()


upp_frame.mainloop()