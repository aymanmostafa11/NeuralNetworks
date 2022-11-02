import tkinter as tk
import matplotlib.pyplot as plt
from DataManager import get_features

ROWS = {'FEATURES': 0, 'CLASSES': 1, "MISC": 2}
WINDOW_SIZE = "800x600"
# TODO: Create GUI
main_window = tk.Tk()


main_frame = tk.Frame(main_window)
main_frame.pack(fill="both", expand=1)

results_frame = tk.Frame(main_window)


def run_gui():
    initialize_window(main_window)

    frames = initialize_frames(main_frame)

    tk.Label(frames['FEATURES'], text="f1", fg="red").grid(row=0, column=0)
    tk.Label(frames['CLASSES'], text="f1", fg="red").grid(row=0, column=0)
    tk.Label(frames["MISC"], text="hhh", fg="red").grid(row=0, column=0)

    main_window.mainloop()


def initialize_window(window):
    window.geometry(WINDOW_SIZE)
    window.title('NN Task')

    row_weight = int(100 / len(ROWS))
    for name, row_index in ROWS.items():
        main_frame.rowconfigure(row_index, weight=row_weight)


def initialize_frames(parent_frame: tk.Frame):
    frames = {}
    for name, row_index in ROWS.items():
        frames[name] = tk.Frame(parent_frame, highlightbackground="black", highlightthickness=1)
        frames[name].grid(row=row_index, column=0)
    return frames


#
# def initialize_starting_frame(window):
#     main_frame.pack(fill="both", expand=1)
#     tk.Label(main_frame, text="elSayadAi.com", fg="red", width=30).grid(0, 0)
#     btn = tk.Button(main_frame, text='Click me',
#                     width=20, command=lambda:change_frame(results_frame, main_frame)).place(x=40, y=50)
#
# def initialize_results_frame(window):
#     tk.Label(results_frame, text="of", fg="red", width=30).place(x=10, y = 10)
#
# def change_frame(x, y):
#     x.pack(fill="both", expand=1)
#     y.forget()

#run_gui()

