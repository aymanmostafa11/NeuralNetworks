import tkinter as tk
import tkinter.ttk as ttk
import tkinter.messagebox

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import seaborn as sns

import gui.logic  # to access variables
from gui.logic import *
from gui.util import valid_input, ChecklistBox, switch_frames


class WidgetManager:
    __WINDOW_SIZE = "800x600"
    __WINDOW_TITLE = "NN Task"
    __ROWS = {name: index for index, name in enumerate(['MODEL', 'FEATURES', 'CLASSES', 'PARAMETERS', 'BUTTONS'])}
    FEATURES = ('bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'gender', 'body_mass_g')
    CLASSES = ('Adelie', 'Gentoo', 'Chinstrap')
    AVAILABLE_MODELS = ["Perceptron", "Adaline", "MLP"]
    AVAILABLE_ACTIVATIONS = ["Sigmoid", "Tanh"]

    def __init__(self):
        self.main_window = tk.Tk()
        self.main_frame = tk.Frame(self.main_window)
        self.main_frame.pack(fill="y", expand=1)

        self.frames: dict[str: tk.Frame, ...] = {}
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

        self.frames["ARCHI"] = tk.Frame(self.main_frame, highlightbackground="black", highlightthickness=1,
                                        name="archi")

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
                                            *self.AVAILABLE_MODELS, command=self.switch_model)
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
        self.hyper_parameters_widgets['bias'] = tk.BooleanVar(value=True)
        tk.Checkbutton(parent_frame, variable=self.hyper_parameters_widgets['bias'], text="Bias",
                       onvalue=True, offvalue=False,
                       relief="flat", highlightthickness=0).grid(row=0, column=4, padx=10)

        # Min Threshold (Adaline)
        default_val = "1.0"
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

        params = {'lr': float(self.hyper_parameters_widgets['lr'].get()),
                  'epochs': int(self.hyper_parameters_widgets['epochs'].get()),
                  'bias': self.hyper_parameters_widgets['bias'].get()}
        if self.selected_model == "Adaline":
            params["min_threshold"] = float(self.hyper_parameters_widgets["min_threshold"].get())
        elif self.selected_model == "MLP":
            archi_text = self.hyper_parameters_widgets["archi"].get().split(",")
            params["archi"] = [int(layer_neurons) for layer_neurons in archi_text]
            params["activation"] = self.hyper_parameters_widgets["activation_val"].get()

        return params

    def __verify_parameters(self):
        model = self.selected_model

        if model == "Adaline" or model == "Perceptron":
            classes = self.get_selected_classes()
            features = self.get_selected_features()
            if len(features) != 2 or len(classes) != 2:
                tk.messagebox.showerror(title="Invalid Parameters",
                                        message="Error in number of features or classes selected\n"
                                                "Please select exactly 2 features and 2 classes")
                return False

        try:
            lr = float(self.hyper_parameters_widgets["lr"].get())
            epochs = int(self.hyper_parameters_widgets["epochs"].get())
            assert lr > 0 and epochs > 0

            if model == "Adaline":
                min_thresh = float(self.hyper_parameters_widgets["min_threshold"].get())
                assert min_thresh >= 0

            elif model == "MLP":
                archi_text = self.hyper_parameters_widgets["archi"].get().split(",")
                [int(i) for i in archi_text]

            return True
        except ValueError or AssertionError:
            error_msg = "Learning rate should be between 0 and 1,\nEpochs should be an integer > 0"
            if model == "Adaline":
                error_msg += "\nMin Threshold should be a float > 0"
            elif model == "MLP":
                error_msg += "\nNetwork configuration should be in the form 'l1_neurons,l2_neurons,etc..'"

            tk.messagebox.showerror("Invalid Parameters", error_msg)
            return False

    def __reset_parameters_widgets(self):
        if "min_threshold" in self.hyper_parameters_widgets.keys():
            self.hyper_parameters_widgets["min_threshold_label"].grid_forget()
            self.hyper_parameters_widgets["min_threshold"].grid_forget()

        if "activation" in self.hyper_parameters_widgets.keys():
            self.hyper_parameters_widgets["activation_label"].grid_forget()
            self.hyper_parameters_widgets["activation"].grid_forget()

    ###########
    ### Buttons
    ###########
    def train_button(self):
        if not self.__verify_parameters():
            return

        features = self.FEATURES
        classes = self.CLASSES
        if self.selected_model == "Adaline" or self.selected_model == "Perceptron":
            features = self.get_selected_features()
            classes = self.get_selected_classes()


        # fit model
        fit_model(self.selected_model, self.get_hyperparameters(), features, classes)

        if self.selected_model != "MLP":
            tk.messagebox.showinfo(title="Model Fitted", message=f"Model Finished Fitting with train accuracy "
                                                                 f"{test_model(train_only=True)}")

        # enable other buttons
        self.buttons['test'].config(state=tk.NORMAL)
        self.buttons['plot'].config(state=tk.NORMAL)

    def test_button(self):
        classes = self.get_selected_classes()
        is_mlp = False
        if self.selected_model == "MLP":
            classes = self.CLASSES
            is_mlp = True


        train_eval, test_eval, conf_mat = test_model(classes, mlp=is_mlp)
        if is_mlp:
            tk.messagebox.showinfo("Model Tested",
                                   f"Model MSE on train data {train_eval}\n"
                                   f"Model MSE on test data {test_eval}\n\n"
                                   f"Check Console for confusion matrix!")
        else:
            tk.messagebox.showinfo("Model Tested",
                                   f"Model Accuracy on train data {train_eval}\n"
                                   f"Model Accuracy on test data {test_eval}\n\n"
                                   f"Check Console for confusion matrix!")

    def plot_button(self):
        self.init_viz_frame()
        switch_frames(widget_manager.viz_frame, widget_manager.main_frame, destroy=False)

    def switch_model(self, selection):
        self.selected_model = selection

        if selection == "Perceptron":
            self.frames["FEATURES"].grid(row=1, column=0)
            self.frames["CLASSES"].grid(row=2, column=0)
            self.buttons['plot'].grid(row=0, column=2)
            self.frames["ARCHI"].grid_forget()
            if "activation" in self.hyper_parameters_widgets.keys():
                self.hyper_parameters_widgets["activation"].grid_forget()
                self.hyper_parameters_widgets["activation_label"].grid_forget()
            self.__reset_parameters_widgets()

        elif selection == "Adaline":
            self.frames["FEATURES"].grid(row=1, column=0)
            self.frames["CLASSES"].grid(row=2, column=0)
            self.buttons['plot'].grid(row=0, column=2)
            self.frames["ARCHI"].grid_forget()

            if "activation" in self.hyper_parameters_widgets.keys():
                self.hyper_parameters_widgets["activation"].grid_forget()
                self.hyper_parameters_widgets["activation_label"].grid_forget()
            self.__reset_parameters_widgets()
            # TODO: add logic for only epochs or min_threshold can be provided not both
            column_count = self.frames["PARAMETERS"].grid_size()[0]
            column_count += 1
            self.hyper_parameters_widgets["min_threshold_label"].grid(row=0, column=column_count)
            column_count += 1
            self.hyper_parameters_widgets["min_threshold"].grid(row=0, column=column_count, padx=5)

        elif selection == "MLP":
            self.__reset_parameters_widgets()
            self.show_mlp_gui()

    def show_mlp_gui(self):
        self.frames["FEATURES"].grid_forget()
        self.frames["CLASSES"].grid_forget()
        self.buttons['plot'].grid_forget()

        self.frames["ARCHI"].grid(row=1, column=0, rowspan=2)

        tk.Label(self.frames["ARCHI"], text="Choose network configurations: ").grid(row=0, column=0)
        archi_text = tk.StringVar(self.frames["ARCHI"], "5, 5, 3")
        tk.Entry(self.frames["ARCHI"], width=20,
                 textvariable=archi_text).grid(row=1, column=0)
        self.hyper_parameters_widgets["archi"] = archi_text

        self.hyper_parameters_widgets["activation_label"] = tk.Label(self.frames["PARAMETERS"],
                                                                     text="Choose activation: ")
        self.hyper_parameters_widgets["activation_label"].grid(row=1, column=0)

        activation_val = tk.StringVar(self.frames["ARCHI"])
        self.hyper_parameters_widgets["activation"] = ttk.OptionMenu(self.frames["PARAMETERS"],
                                                                     activation_val,
                                                                     self.AVAILABLE_ACTIVATIONS[0],
                                                                     *self.AVAILABLE_ACTIVATIONS)
        self.hyper_parameters_widgets["activation_val"]: tk.StringVar = activation_val
        self.hyper_parameters_widgets["activation"].grid(row=1, column=1)




widget_manager = WidgetManager()


def run_gui():
    widget_manager.main_window.mainloop()

