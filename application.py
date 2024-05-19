from customtkinter import *

# variables
HEIGHT = 800
WIDTH = 800
FRAME_WIDTH = 750
FRAME_HEIGHT = 400
TEXT_FRAME_HEIGHT = 300

desc_text = "lorem ipsum something something yada yada yada"

# setting up the window

set_appearance_mode("System")
set_default_color_theme("blue")

app = CTk()
app.geometry(f"{WIDTH}x{HEIGHT}")

# setting up the frame

textFrame = CTkFrame(master=app, width=FRAME_WIDTH, height=TEXT_FRAME_HEIGHT, border_width=1, border_color="red")
textFrame.pack(expand=True)
textFrame.place(relx=0.5, rely=0.225, anchor=CENTER)

btnFrame = CTkFrame(master=app, width=FRAME_WIDTH, height=FRAME_HEIGHT, border_width=1, border_color="blue")
btnFrame.pack(expand=True)
btnFrame.place(relx=0.5, rely=0.7, anchor=CENTER)
# functions

def button_clicked(num):
    print(f"Button {num} clicked")


# adding the labels
heading = CTkLabel(master=textFrame, text="Human Posture Detection")
heading.place(relx=0.5, rely=0.2, anchor=CENTER)

desc = CTkLabel(master=textFrame, text=desc_text)
desc.place(relx=0.4, rely=0.4, anchor=E)


# adding the buttons
button1 = CTkButton(master=btnFrame, text="Posture Detection", command=lambda: button_clicked(1))
button1.place(relx=0.5, rely=0.5, anchor=CENTER)

button2 = CTkButton(master=btnFrame, text="Yoga Pose Detection", command=lambda: button_clicked(2))
button2.place(relx=0.25, rely=0.5, anchor=CENTER)

button3 = CTkButton(master=btnFrame, text="Gym Pose Detection", command=lambda: button_clicked(3))
button3.place(relx=0.75, rely=0.5, anchor=CENTER)

if __name__ == "__main__":
    app.mainloop()