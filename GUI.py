import tkinter as tk
import matplotlib.pyplot as plt


# TODO: Create GUI
window = tk.Tk()
Starting_frame = tk.Frame(window)
results_frame = tk.Frame(window)

def run_gui():
    initialize_window(window)
    initialize_starting_frame(window)
    initialize_results_frame(window)
    window.mainloop()

def initialize_window(window):
    window.geometry("900x400")
    window.title('NN Task')


def initialize_starting_frame(window):
    Starting_frame.pack(fill="both", expand=1)
    tk.Label(Starting_frame, text="elSayadAi.com", fg="red", width=30).place(x=10, y = 10)
    btn = tk.Button(Starting_frame, text='Click me',
                              width=20, command=lambda:change_frame(results_frame,Starting_frame)).place(x=40,y=50)

def initialize_results_frame(window):
    tk.Label(results_frame, text="of", fg="red", width=30).place(x=10, y = 10)

def change_frame(x, y):
    x.pack(fill="both", expand=1)
    y.forget()

run_gui()

