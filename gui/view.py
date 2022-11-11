import tkinter as tk
import tkinter.ttk as ttk
import tkinter.messagebox

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import seaborn as sns

import gui.logic  # to access variables
from gui.logic import *
from gui.util import valid_input, ChecklistBox, switch_frames


# TODO : fix buttons command inside class can't see functions outside
class WidgetManager:
    __WINDOW_SIZE = "800x600"
    __WINDOW_TITLE = "NN Task"
    __ROWS = {name: index for index, name in enumerate(['MODEL', 'FEATURES', 'CLASSES', 'PARAMETERS', 'BUTTONS'])}
    FEATURES = ('bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'gender', 'body_mass_g')
    CLASSES = ('Adelie', 'Gentoo', 'Chinstrap')
    AVAILABLE_MODELS = ["Perceptron", "Adaline", "MLP"]

    def __init__(self):
        self.main_window = tk.Tk()
        self.main_frame = tk.Frame(self.main_window)
        self.main_frame.pack(fill="y", expand=1)

        self.frames = {}
        self.viz_frame = None

        self.check_lists = {}
        self.buttons = {}
        self.hyper_parameters_widgets = {}
        self.model_listbox: tk.Listbox = None

        self.selected_model = self.AVAILABLE_MODELS[0]

        self.__init_window()
        self.__init_frames()

    def __init_window(self):
        self.main_window.geometry(self.__WINDOW_SIZE)
        self.main_window.title(self.__WINDOW_TITLE)
        row_weight = 100 // len(self.__ROWS)
        for name, row_index in self.__ROWS.items():
            self.main_frame.rowconfigure(row_index, weight=row_weight)

    def __init_frames(self):
        # create frames
        for name, row_index in self.__ROWS.items():
            self.frames[name] = tk.Frame(self.main_frame, highlightbackground="black", highlightthickness=1,
                                         name=str.lower(name))
            self.frames[name].grid(row=row_index, column=0)

        # init
        self.__init_model_frame()
        self.__init_classes_frame()
        self.__init_features_frame()
        self.__init_parameters_frame()
        self.__init_buttons_frame()

    def __init_model_frame(self):
        parent_frame = self.frames["MODEL"]

        tk.Label(parent_frame, text="Choose a model: ").grid(row=0, column=0)
        self.model_listbox = ttk.OptionMenu(parent_frame, tk.StringVar(parent_frame), self.AVAILABLE_MODELS[0],
                                            *self.AVAILABLE_MODELS)
        self.model_listbox.grid(row=0, column=1)

    def __init_features_frame(self):
        parent_frame = self.frames["FEATURES"]
        tk.Label(parent_frame, text="Choose two features: ").grid(row=0, column=0)
        self.check_lists['features'] = ChecklistBox(parent_frame, self.FEATURES, Side='top')
        self.check_lists['features'].grid(row=0, column=1, padx=50)

    def __init_classes_frame(self):
        parent_frame = self.frames["CLASSES"]
        tk.Label(parent_frame, text="Choose two classes: ").grid(row=0, column=0)
        self.check_lists['classes'] = ChecklistBox(parent_frame, self.CLASSES, Side='top')
        self.check_lists['classes'].grid(row=0, column=1, padx=50)

    def __init_parameters_frame(self):
        parent_frame = self.frames["PARAMETERS"]
        text_box_width = 5
        text_font = 'Arial 12'

        parent_frame.grid_rowconfigure(10, weight=1)
        parent_frame.grid_columnconfigure(10, weight=1)

        # Learning Rate
        default_val = "0.005"
        tk.Label(parent_frame, text="Enter learning rate: ").grid(row=0, column=0)
        self.hyper_parameters_widgets['lr'] = tk.Entry(parent_frame, width=text_box_width, font=text_font,
                                                       textvariable=tk.StringVar(parent_frame, default_val))
        self.hyper_parameters_widgets['lr'].grid(row=0, column=1, padx=10)

        # Epochs
        default_val = "100"
        tk.Label(parent_frame, text="Epochs: ").grid(row=0, column=2)
        self.hyper_parameters_widgets['epochs'] = tk.Entry(parent_frame, width=text_box_width, font=text_font,
                                                           textvariable=tk.StringVar(parent_frame, default_val))
        self.hyper_parameters_widgets['epochs'].grid(row=0, column=3, padx=10)

        # bias
        self.hyper_parameters_widgets['bias'] = tk.BooleanVar(value=False)
        tk.Checkbutton(parent_frame, variable=self.hyper_parameters_widgets['bias'], text="Bias",
                       onvalue=True, offvalue=False,
                       anchor="w", width=10,
                       relief="flat", highlightthickness=0).grid(row=0, column=4, padx=10)

        # Min Threshold (Adaline)
        # TODO: add logic to hide and show depending on selected model
        default_val = "10"
        self.hyper_parameters_widgets["min_threshold_label"] = tk.Label(parent_frame, text="Min Threshold: ")
        self.hyper_parameters_widgets['min_threshold'] = tk.Entry(parent_frame, width=text_box_width, font=text_font,
                                                                  textvariable=tk.StringVar(parent_frame, default_val))

    def __init_buttons_frame(self):
        parent_frame = self.frames["BUTTONS"]
        self.buttons['train'] = ttk.Button(parent_frame, text="Train", width=20, command=self.train_button)
        self.buttons['train'].grid(row=0, column=0, padx=20)

        self.buttons['test'] = ttk.Button(parent_frame, text="Test", width=20, command=self.test_button,
                                          state=tk.DISABLED)
        self.buttons['test'].grid(row=0, column=1, padx=20)

        self.buttons['plot'] = ttk.Button(parent_frame, text="Plot", width=20, command=self.plot_button,
                                          state=tk.DISABLED)
        self.buttons['plot'].grid(row=0, column=3, padx=20)

    def init_viz_frame(self):

        # created here because it's destroyed when closing the plotting window
        self.viz_frame = tk.Frame(self.main_window)

        data = get_viz_data(self.get_selected_classes(),
                            self.get_selected_features())

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

        canvas = FigureCanvasTkAgg(fig, master=self.viz_frame)
        canvas.draw()
        canvas.get_tk_widget().pack()

        # return button
        tk.Button(self.viz_frame, text="return",
                  command=lambda: switch_frames(self.main_frame, self.viz_frame)).pack()

    def get_selected_classes(self):
        return self.check_lists["classes"].getCheckedItems()

    def get_selected_features(self):
        return self.check_lists["features"].getCheckedItems()

    def get_hyperparameters(self):
        # TODO : perform value checks
        return {'lr': float(self.hyper_parameters_widgets['lr'].get()),
                'epochs': int(self.hyper_parameters_widgets['epochs'].get()),
                'bias': self.hyper_parameters_widgets['bias'].get()}

    ###########
    ### Buttons
    ###########
    def train_button(self):
        selected_classes = self.get_selected_classes()
        selected_features = self.get_selected_features()

        if not valid_input(selected_features, selected_classes, self.hyper_parameters_widgets):
            return

        # fit model
        fit_model(selected_features, selected_classes, self.get_hyperparameters())
        tk.messagebox.showinfo(title="Model Fitted", message=f"Model Finished Fitting with train accuracy "
                                                             f"{test_model(train_only=True)}")
        # enable other buttons
        self.buttons['test'].config(state=tk.NORMAL)
        self.buttons['plot'].config(state=tk.NORMAL)

    def test_button(self):
        train_acc, test_acc, conf_mat = test_model(widget_manager.get_selected_classes())
        tk.messagebox.showinfo("Model Tested",
                               f"Model Accuracy on train data {train_acc}\n"
                               f"Model Accuracy on test data {test_acc}\n\n"
                               f"Check Console for confusion matrix!")

    def plot_button(self):
        self.init_viz_frame()
        switch_frames(widget_manager.viz_frame, widget_manager.main_frame, destroy=False)

    def switch_model(self, selection):
        self.selected_model = selection

        # TODO: Update options according to model
        if selection in ["Adaline", "MLP"]:
            tk.messagebox.showinfo("Model Not Available", "This model hasn't been added yet")


widget_manager = WidgetManager()


def run_gui():
    widget_manager.main_window.mainloop()
