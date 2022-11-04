import tkinter as tk
import tkinter.messagebox

import matplotlib.pyplot as plt
from gui_logic import *

ROWS = {'FEATURES': 0, 'CLASSES': 1, "MISC": 2, "BUTTONS": 3}
WINDOW_SIZE = "800x600"
FEATURES = ('bill_length', 'bill_depth', 'flipper_length', 'gender', 'body_mass')
CLASSES = ('C1', 'C2', 'C3')
check_lists = {}
buttons = {}
hyper_parameters = {}
help_message = None

main_window = tk.Tk()
main_frame = tk.Frame(main_window)
main_frame.pack(fill="both", expand=1)
results_frame = tk.Frame(main_window)


# TODO: add show visualization logic

def run_gui():
    initialize_window(main_window)
    frames = initialize_frames(main_frame)
    initialize_features_frame(frames['FEATURES'])
    initialize_classes_frame(frames['CLASSES'])
    initialize_misc_frame(frames['MISC'])
    initialize_buttons_frame(frames['BUTTONS'])

    # help_message = tk.Label(main_frame, text= "Help Text", foreground="red")
    # help_message.grid(row=len(ROWS) + 1, column=0)

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
        frames[name] = tk.Frame(parent_frame, highlightbackground="black", highlightthickness=1, name=str.lower(name))
        frames[name].grid(row=row_index, column=0)
    return frames


def initialize_features_frame(features_frame: tk.Frame):
    tk.Label(features_frame, text="Choose two features: ").grid(row=0, column=0)
    check_lists['features'] = ChecklistBox(features_frame, FEATURES, Side='top')
    check_lists['features'].grid(row=0, column=1, padx=50)


def initialize_classes_frame(classes_frame: tk.Frame):
    tk.Label(classes_frame, text="Choose two classes: ").grid(row=0, column=0)
    check_lists['classes'] = ChecklistBox(classes_frame, CLASSES, Side='top')
    check_lists['classes'].grid(row=0, column=1, padx=50)


def initialize_misc_frame(misc_frame: tk.Frame):
    text_box_width = 5

    misc_frame.grid_rowconfigure(10, weight=1)
    misc_frame.grid_columnconfigure(10, weight=1)

    tk.Label(misc_frame, text="Enter learning rate: ").grid(row=0, column=0)
    hyper_parameters['lr'] = tk.Entry(misc_frame, width=text_box_width, font='Arial 14')
    hyper_parameters['lr'].grid(row=0, column=1, padx=10)

    tk.Label(misc_frame, text="Epochs: ").grid(row=0, column=2)
    hyper_parameters['epochs'] = tk.Entry(misc_frame, width=text_box_width, font='Arial 14')
    hyper_parameters['epochs'].grid(row=0, column=3, padx=10)

    var = tk.BooleanVar(value=False)
    hyper_parameters['bias'] = tk.Checkbutton(misc_frame, var=var, text="Bias",
                                              onvalue=True, offvalue=False,
                                              anchor="w", width=10,
                                              relief="flat", highlightthickness=0)
    hyper_parameters['bias'].grid(row=0, column=4, padx=10)


def initialize_buttons_frame(buttons_frame: tk.Frame):
    buttons['train'] = tk.Button(buttons_frame, text="Train", command=train_button)
    buttons['train'].grid(row=0, column=0, padx=20)

    buttons['test'] = tk.Button(buttons_frame, text="Test", command=test_button, state=tk.DISABLED)
    buttons['test'].grid(row=0, column=1, padx=20)

    buttons['retrain'] = tk.Button(buttons_frame, text="reTrain", command=retrain_button, state=tk.DISABLED)
    buttons['retrain'].grid(row=0, column=2, padx=20)


####################
# Buttons
####################
def train_button():
    # verify data
    choosen_features = check_lists['features'].getCheckedItems()
    choosen_classes = check_lists['classes'].getCheckedItems()

    if not valid_input(choosen_features, choosen_classes):
        return

    # fit model
    fit_model(choosen_features, choosen_classes, hyper_parameters)
    tk.messagebox.showinfo(title="Model Fitted", message="Model Finished Fitting")

    # enable other buttons
    buttons['test'].config(state=tk.NORMAL)
    buttons['retrain'].config(state=tk.NORMAL)


def test_button():
    test_model()


def retrain_button():
    retrain_model()


def valid_input(features, classes):
    if len(features) != 2 or len(classes) != 2:
        tk.messagebox.showerror(title="Invalid Parameters",
                                message="Error in number of features or classes selected, pleases select exactly 2 features and 2 classes")
        return 0
    return 1


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
