import tkinter as tk
import matplotlib.pyplot as plt
from DataManager import get_features

ROWS = {'FEATURES': 0, 'CLASSES': 1, "MISC": 2,"BUTTONS": 3}
WINDOW_SIZE = "800x600"
FEATURES = ('bill_length','bill_depth','flipper_length','gender','body_mass')
CLASSES = ('C1','C2','C3')
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
    tk.Label(frames["BUTTONS"], text="hhh", fg="red").grid(row=0, column=0)
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
    features_checklist=ChecklistBox(features_frame,FEATURES,Side='top')
    features_checklist.grid(row=0, column=1,padx=50)
    

def initialize_classes_frame(classes_frame: tk.Frame):
    tk.Label(classes_frame, text="Choose two classes: ").grid(row=0, column=0)
    classes_checklist=ChecklistBox(classes_frame,CLASSES,Side='top')
    classes_checklist.grid(row=0, column=1,padx=50)


def initialize_misc_frame(misc_frame: tk.Frame):
    tk.Label(misc_frame, text="Enter learning rate: ").grid(row=0, column=0)
    tk.Entry(misc_frame, width=10, font='Arial 14').grid(row=0, column=1)

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
            value =  var.get()
            if value:
                values.append(value)
        print(values)
        return values
run_gui()

