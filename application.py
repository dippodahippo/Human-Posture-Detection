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

def button_clicked():
    print("yo mama")

# adding the button
button = CTkButton(master=app, text="Test", command=button_clicked)
button.place(relx=0.5, rely=0.5, anchor=CENTER)

app.mainloop()