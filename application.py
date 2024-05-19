from customtkinter import *

# variables
HEIGHT = 600
WIDTH = 600

# setting up the window

set_appearance_mode("System")
set_default_color_theme("blue")

app = CTk()
app.geometry(f"{WIDTH}x{HEIGHT}")

# functions

def button_clicked(num):
    print(f"{num} clicked")

# adding the button
button1 = CTkButton(master=app, text="Posture Detection", command=lambda: button_clicked(1))
button1.place(relx=0.5, rely=0.5, anchor=CENTER)

button2 = CTkButton(master=app, text="Yoga Pose Detection", command=lambda: button_clicked(2))
button2.place(relx=0.25, rely=0.5, anchor=CENTER)

button3 = CTkButton(master=app, text="Gym Pose Detection", command=lambda: button_clicked(3))
button3.place(relx=0.75, rely=0.5, anchor=CENTER)

app.mainloop()