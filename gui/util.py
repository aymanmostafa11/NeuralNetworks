import tkinter as tk


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
