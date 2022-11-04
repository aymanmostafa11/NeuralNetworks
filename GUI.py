import tkinter as tk
import matplotlib.pyplot as plt
from gui_logic import *

ROWS = {'FEATURES': 0, 'CLASSES': 1, "MISC": 2, "BUTTONS": 3}
WINDOW_SIZE = "800x600"
FEATURES = ('bill_length', 'bill_depth', 'flipper_length', 'gender', 'body_mass')
CLASSES = ('C1', 'C2', 'C3')
# TODO: Create GUI

main_window = tk.Tk()
main_frame = tk.Frame(main_window)
main_frame.pack(fill="both", expand=1)
results_frame = tk.Frame(main_window)


def run_gui():
    initialize_window(main_window)
    frames = initialize_frames(main_frame)
    initialize_features_frame(frames['FEATURES'])
    initialize_classes_frame(frames['CLASSES'])
    initialize_misc_frame(frames['MISC'])
    initialize_buttons_frame(frames['BUTTONS'])

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


def initialize_features_frame(features_frame: tk.Frame):
    tk.Label(features_frame, text="Choose two features: ").grid(row=0, column=0)
    features_checklist = ChecklistBox(features_frame, FEATURES, Side='top')
    features_checklist.grid(row=0, column=1, padx=50)


def initialize_classes_frame(classes_frame: tk.Frame):
    tk.Label(classes_frame, text="Choose two classes: ").grid(row=0, column=0)
    classes_checklist = ChecklistBox(classes_frame, CLASSES, Side='top')
    classes_checklist.grid(row=0, column=1, padx=50)


def initialize_misc_frame(misc_frame: tk.Frame):
    text_box_width = 5

    misc_frame.grid_rowconfigure(10, weight=1)
    misc_frame.grid_columnconfigure(10, weight=1)

    tk.Label(misc_frame, text="Enter learning rate: ").grid(row=0, column=0)
    tk.Entry(misc_frame, width=text_box_width, font='Arial 14').grid(row=0, column=1, padx=10)

    tk.Label(misc_frame, text="Epochs: ").grid(row=0, column=2)
    tk.Entry(misc_frame, width=text_box_width, font='Arial 14').grid(row=0, column=3, padx=10)

    var = tk.BooleanVar(value=False)
    tk.Checkbutton(misc_frame, var=var, text="Bias",
                   onvalue=True, offvalue=False,
                   anchor="w", width=10,
                   relief="flat", highlightthickness=0).grid(row=0, column=4, padx=10)


def initialize_buttons_frame(buttons_frame: tk.Frame):
    train = tk.Button(buttons_frame, text="Train", command=train_button)
    train.grid(row=0, column=0, padx=20)

    test = tk.Button(buttons_frame, text="Test", command=test_button, state=tk.DISABLED)
    test.grid(row=0, column=1, padx=20)

    retrain = tk.Button(buttons_frame, text="reTrain", command=retrain_button, state=tk.DISABLED)
    retrain.grid(row=0, column=2, padx=20)


class ChecklistBox(tk.Frame):
    def __init__(self, parent, choices, Side, **kwargs):
        tk.Frame.__init__(self, parent, **kwargs)
        self.vars = []
        bg = self.cget("background")
        for choice in choices:
            var = tk.StringVar(value=choice)
            self.vars.append(var)
            cb = tk.Checkbutton(self, var=var, text=choice,
                                onvalue=choice, offvalue="",
                                anchor="w", width=10, background=bg,
                                relief="flat", highlightthickness=0
                                )
            cb.deselect()
            cb.pack(side=Side, fill="x", anchor="w")

    def getCheckedItems(self):
        values = []
        for var in self.vars:
            value = var.get()
            if value:
                values.append(value)
        print(values)
        return values


run_gui()
