import matplotlib.pyplot as plt
import tkinter as tk
import tkinter.messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import seaborn as sns
import numpy as np
from gui_logic import *
import gui_logic

ROWS = {'FEATURES': 0, 'CLASSES': 1, "MISC": 2, "BUTTONS": 3}
WINDOW_SIZE = "800x600"
FEATURES = ('bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'gender', 'body_mass_g')
CLASSES = ('Adelie', 'Gentoo', 'Chinstrap')
check_lists = {}
buttons = {}
hyper_parameters_widgets = {}

main_window = tk.Tk()
main_frame = tk.Frame(main_window)
visualization_frame: tk.Frame
main_frame.pack(fill="both", expand=1)
results_frame = tk.Frame(main_window)


def run_gui():
    initialize_window(main_window)
    frames = initialize_frames(main_frame)
    initialize_features_frame(frames['FEATURES'])
    initialize_classes_frame(frames['CLASSES'])
    initialize_misc_frame(frames['MISC'])
    initialize_buttons_frame(frames['BUTTONS'])

    help_message = tk.Label(main_frame,
                            text="Don't forget to Train the model after changing parameters to get updated results",
                            foreground="grey")
    help_message.grid(row=len(ROWS) + 1, column=0)

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
    check_lists['features'] = ChecklistBox(features_frame, FEATURES, Side='top', )
    check_lists['features'].grid(row=0, column=1, padx=50)


def initialize_classes_frame(classes_frame: tk.Frame):
    tk.Label(classes_frame, text="Choose two classes: ").grid(row=0, column=0)
    check_lists['classes'] = ChecklistBox(classes_frame, CLASSES, Side='top')
    check_lists['classes'].grid(row=0, column=1, padx=50)


def initialize_misc_frame(misc_frame: tk.Frame):
    text_box_width = 5
    text_font = 'Arial 12'

    misc_frame.grid_rowconfigure(10, weight=1)
    misc_frame.grid_columnconfigure(10, weight=1)

    tk.Label(misc_frame, text="Enter learning rate: ").grid(row=0, column=0)
    hyper_parameters_widgets['lr'] = tk.Entry(misc_frame, width=text_box_width, font=text_font,
                                              textvariable=tk.StringVar(misc_frame, "0.005"))
    hyper_parameters_widgets['lr'].grid(row=0, column=1, padx=10)

    tk.Label(misc_frame, text="Epochs: ").grid(row=0, column=2)
    hyper_parameters_widgets['epochs'] = tk.Entry(misc_frame, width=text_box_width, font=text_font,
                                                  textvariable=tk.StringVar(misc_frame, "100"))
    hyper_parameters_widgets['epochs'].grid(row=0, column=3, padx=10)

    hyper_parameters_widgets['bias'] = tk.BooleanVar(value=False)
    tk.Checkbutton(misc_frame, variable=hyper_parameters_widgets['bias'], text="Bias",
                   onvalue=True, offvalue=False,
                   anchor="w", width=10,
                   relief="flat", highlightthickness=0).grid(row=0, column=4, padx=10)


def initialize_buttons_frame(buttons_frame: tk.Frame):
    buttons['train'] = tk.Button(buttons_frame, text="Train",width=20 ,command=train_button)
    buttons['train'].grid(row=0, column=0, padx=20)

    buttons['test'] = tk.Button(buttons_frame, text="Test",width=20 , command=test_button, state=tk.DISABLED)
    buttons['test'].grid(row=0, column=1, padx=20)

    buttons['retrain'] = tk.Button(buttons_frame, text="reTrain", width=20 ,command=retrain_button, state=tk.DISABLED)
    buttons['retrain'].grid(row=0, column=2, padx=20)

    buttons['plot'] = tk.Button(buttons_frame, text="Plot", width=20, command=plot_button, state=tk.DISABLED)
    buttons['plot'].grid(row=0, column=3, padx=20)


def initialize_Visualization_frame(data):
    global visualization_frame
    visualization_frame = tk.Frame(main_window)
    # the figure that will contain the plot
    fig = Figure(figsize=(5, 5), dpi=100)

    X, Y = data.drop(["species"], axis=1), data['species']
    w = get_model_weights()

    # adding the subplot
    ax = fig.add_subplot(111)
    sns.scatterplot(x=X.iloc[:, 0], y=X.iloc[:, 1], hue=Y, ax=ax)

    # Calculate boundary line
    x_vals = np.array(ax.get_xlim())
    y_vals = - (x_vals * w[1] + w[0]) / w[2]

    # draw boundary line
    ax.plot(x_vals, y_vals, c='red')

    canvas = FigureCanvasTkAgg(fig, master=visualization_frame)
    canvas.draw()
    canvas.get_tk_widget().pack()

    # return button
    tk.Button(visualization_frame, text="return", command=lambda: switch_frames(main_frame, visualization_frame)).pack()


####################
# Buttons
####################
def train_button():
    # verify data
    choosen_features = check_lists['features'].getCheckedItems()
    choosen_classes = check_lists['classes'].getCheckedItems()
    #choosen_features = ['bill_length_mm', "bill_depth_mm"]
    #choosen_classes = ["Adelie", "Gentoo"]

    if not valid_input(choosen_features, choosen_classes, hyper_parameters_widgets):
        return

    # fit model
    fit_model(choosen_features, choosen_classes, {'lr': float(hyper_parameters_widgets['lr'].get()),
                                                  'epochs': int(hyper_parameters_widgets['epochs'].get()),
                                                  'bias': hyper_parameters_widgets['bias'].get()})
    tk.messagebox.showinfo(title="Model Fitted", message=f"Model Finished Fitting with train accuracy {test_model(True)}")
    # enable other buttons
    buttons['test'].config(state=tk.NORMAL)
    buttons['retrain'].config(state=tk.NORMAL)
    buttons['plot'].config(state=tk.NORMAL)


def test_button():
    train_acc, test_acc = test_model()
    tk.messagebox.showinfo("Model Tested",
                           f"Model Accuracy on train data {train_acc}\n"
                           f"Model Accuracy on test data {test_acc}")


def retrain_button():
    #retrain_model()
    tk.messagebox.showinfo("Retrain Model", "Coming Soon :D")


def plot_button():
    initialize_Visualization_frame(gui_logic.viz_data)
    switch_frames(visualization_frame, main_frame, destroy=False)


##################
#### Helpers  ####
##################

def valid_input(features, classes, hyper_parameters):
    if len(features) != 2 or len(classes) != 2:
        tk.messagebox.showerror(title="Invalid Parameters",
                                message="Error in number of features or classes selected\n"
                                        "Please select exactly 2 features and 2 classes")
        return 0

    try:
        float(hyper_parameters['lr'].get())
        int(hyper_parameters['epochs'].get())
    except:
        tk.messagebox.showerror(title="Invalid Parameters",
                                message="Learning rate should be between 0 and 1,\nEpochs should be an integer > 0")
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
            cb = tk.Checkbutton(self, variable=var, text=choice,
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
        return values


def switch_frames(to_focus: tk.Frame, to_forget: tk.Frame, destroy=True):
    to_forget.forget()
    if destroy:
        to_forget.destroy()

    to_focus.pack(fill="both", expand=1)

