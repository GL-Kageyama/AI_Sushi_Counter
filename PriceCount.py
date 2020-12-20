# Copyright (c) 2020 a1kageyama
# Released under the MIT license

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageTk
import tkinter as tk
import tkinter.font as font
import glob
import CnnModel
import keras

root = tk.Tk()
root.title("AI Sushi Counter")
root.geometry("780x580")
canvas = tk.Canvas(root, width=780, height=580, bg="#ffffff")
canvas.place(x=0, y=0)
fontSushi = font.Font(family="Times", size=16)
fontPrice = font.Font(family="Times", size=48)

imgRow = 60
imgCol = 60
imgColor = 3
inShape = (imgRow, imgCol, imgColor)
nbClass = 15
totalPrice = 0

# Model Labels
LABELS = ["Bonito", "CaliforniaRoll", "CongerEel", "FattyTuna", "InariSushi", 
          "LeanTuna", "Mackerel", "Salmon", "SalmonRoe", "Scallop", 
          "SeaUrchin", "Shrimp", "SushiBurrito", "Tamago", "TunaSushiRoll"]
# Initial value of Count
COUNT = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
# Sushi Price
PRICE = [2, 3, 4, 5, 2, 2, 2, 2, 4, 3, 4, 3, 3, 1, 1]

# Load the CNN model
model = CnnModel.getModel(inShape, nbClass)
model.load_weights("./SystemData/Model/SushiModel.hdf5")

# Load all the images in the folder
files = glob.glob("./TargetPhoto/*.jpg")


# Predictive Reading Images
def predictTarget(path):

    # Loading Images
    img = Image.open(path)
    img = img.convert("RGB")
    img = img.resize((imgCol, imgRow))

    # Convert to Data
    x = np.asarray(img)
    x = x.reshape(-1, imgRow, imgCol, imgColor)
    x = x / 255

    # Prediction
    pre = model.predict([x])[0]
    idx = pre.argmax()
    per = int(pre[idx] * 100)
    return (idx, per)


# Output Predicted Results
def outputPredict(path):

    idx, per = predictTarget(path)

    # Show Answer
    print(path)
    print(LABELS[idx] + " : " + str(per) + "%")
    print()
    COUNT[idx] += 1


# Preparation for Individual Images
def sushiImage(count, xTxt, yTxt, imgTxt):

    # Character Output
    label = tk.Label(root, text= imgTxt + " × " + str(count), font=fontSushi, background="#ffffff")
    label.place(x=xTxt, y=yTxt)

    # Sushi Icon Retrieval
    if (0 == count):
        img = Image.open(open("SystemData/SushiIcon/0_" + imgTxt + ".png", "rb"))
    elif (0 < count):
        img = Image.open(open("SystemData/SushiIcon/" + imgTxt + ".png", "rb"))

    return img


# Parsing for All Files
for file in files:
    outputPredict(file)


# Calculate the Price from the Count
for i in range(15):
    totalPrice += COUNT[i] * PRICE[i]


# Bonito Image
img1 = sushiImage(COUNT[0], 40, 130, imgTxt="Bonito")
img1 = ImageTk.PhotoImage(img1)
canvas.create_image(0, 0, image=img1, anchor=tk.NW)
# California Roll Image
img2 = sushiImage(COUNT[1], 150, 130, imgTxt="CaliforniaRoll")
img2 = ImageTk.PhotoImage(img2)
canvas.create_image(150, 0, image=img2, anchor=tk.NW)
# Conger Eel Image
img3 = sushiImage(COUNT[2], 310, 130, imgTxt="CongerEel")
img3 = ImageTk.PhotoImage(img3)
canvas.create_image(300, 0, image=img3, anchor=tk.NW)
# Fatty Tuna Image
img4 = sushiImage(COUNT[3], 460, 130, imgTxt="FattyTuna")
img4 = ImageTk.PhotoImage(img4)
canvas.create_image(450, 0, image=img4, anchor=tk.NW)
# Inari Sushi Image
img5 = sushiImage(COUNT[4], 615, 130, imgTxt="InariSushi")
img5 = ImageTk.PhotoImage(img5)
canvas.create_image(600, 0, image=img5, anchor=tk.NW)
# Lean Tuna Image
img6 = sushiImage(COUNT[5], 15, 280, imgTxt="LeanTuna")
img6 = ImageTk.PhotoImage(img6)
canvas.create_image(0, 150, image=img6, anchor=tk.NW)
# Mackerel Image
img7 = sushiImage(COUNT[6], 170, 280, imgTxt="Mackerel")
img7 = ImageTk.PhotoImage(img7)
canvas.create_image(150, 150, image=img7, anchor=tk.NW)
# Salmon Image
img8 = sushiImage(COUNT[7], 330, 280, imgTxt="Salmon")
img8 = ImageTk.PhotoImage(img8)
canvas.create_image(300, 150, image=img8, anchor=tk.NW)
# Salmon Roe Image
img9 = sushiImage(COUNT[8], 460, 280, imgTxt="SalmonRoe")
img9 = ImageTk.PhotoImage(img9)
canvas.create_image(450, 150, image=img9, anchor=tk.NW)
# Scallop Image
img10 = sushiImage(COUNT[9], 630, 280, imgTxt="Scallop")
img10 = ImageTk.PhotoImage(img10)
canvas.create_image(600, 150, image=img10, anchor=tk.NW)
# Sea Urchin Image
img11 = sushiImage(COUNT[10], 20, 430, imgTxt="SeaUrchin")
img11 = ImageTk.PhotoImage(img11)
canvas.create_image(0, 300, image=img11, anchor=tk.NW)
# Shrimp Image
img12 = sushiImage(COUNT[11], 175, 430, imgTxt="Shrimp")
img12 = ImageTk.PhotoImage(img12)
canvas.create_image(150, 300, image=img12, anchor=tk.NW)
# Sushi Burrito Image
img13 = sushiImage(COUNT[12], 310, 430, imgTxt="SushiBurrito")
img13 = ImageTk.PhotoImage(img13)
canvas.create_image(300, 300, image=img13, anchor=tk.NW)
# Tamago Image
img14 = sushiImage(COUNT[13], 480, 430, imgTxt="Tamago")
img14 = ImageTk.PhotoImage(img14)
canvas.create_image(450, 300, image=img14, anchor=tk.NW)
# Tuna Roll Image
img15 = sushiImage(COUNT[14], 620, 430, imgTxt="TunaRoll")
img15 = ImageTk.PhotoImage(img15)
canvas.create_image(600, 300, image=img15, anchor=tk.NW)

# Total Price String
labelPrice = tk.Label(root, text="Total Price  $" + str(totalPrice) + ".00", font=fontPrice, background="#ffffff")
labelPrice.place(x=270, y=500)

root.mainloop()
