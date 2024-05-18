import serial.tools.list_ports
import time
import random

# Setup for the file
ports = serial.tools.list_ports.comports() # this gives back a list of all the ports
serialInst = serial.Serial() # creating an instance of the serial
portsList = [] # list to hold all the ports in a modified format

# Functions

# function to show all the ports and needs to be run for portsList to be populated
def show_ports():
    for port in ports:
        strPort = str(port)
        portsList.append(strPort)
        print(strPort)

# function to select the com port that is connected to the arduino
def select_port(com):
    for i in range(len(portsList)):
        if portsList[i].startswith("COM" + str(com)):
            use = "COM" + str(com)
            return use

# function to send the message to the arduino
def open_port(port):

    # setup the port to send data
    serialInst.baudrate = 9600
    serialInst.port = port
    serialInst.open()
    wait(5)
    return

def send_data(to_be_input):
    # get the data and then send it
    command = str(to_be_input)
    serialInst.write(command.encode('utf-8'))
    return

def close_port():
    serialInst.close()

def wait(seconds=1):
    time.sleep(seconds)
    

if __name__ == "__main__":
    show_ports()
    portNumber = 6
    usablePort = select_port(portNumber)
    open_port(usablePort)
    
    for i in range(10):
        a = random.randint(0,1)
        print(a)
        send_data(a)
    close_port()
