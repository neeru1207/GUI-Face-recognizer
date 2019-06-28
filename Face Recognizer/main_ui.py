import tkinter as tk
from tkinter import font as tkfont
from tkinter import filedialog, messagebox
from cnn_model import CNN
from build_positive import BuildPositiveFaceDataset
from detect_face import DetectFace
from create_test_case import CreateTestSet
from build_negative import BuildNegativeDataset
from load_cnn_from_file import LoadCnn
from PIL import ImageTk, Image
import shutil
names = set()


class MainUI(tk.Tk):

    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)
        global names
        with open("nameslist.txt", "r") as f:
            x = f.read()
            z = x.rstrip().split(" ")
            for i in z:
                names.add(i)
        self.title_font = tkfont.Font(family='Helvetica', size=16, weight="bold")
        self.title("Face Recognizer")
        self.resizable(False, False)
        self.geometry("300x150")
        self.CNNobj = None
        self.Buildposobj = BuildPositiveFaceDataset()
        self.BuildNegobj = BuildNegativeDataset()
        self.BuildNegobj.create_dataset()
        self.createtest = CreateTestSet()
        self.active_name = None
        self.protocol("WM_DELETE_WINDOW", self.on_closing)
        container = tk.Frame(self)
        container.grid(sticky="nsew")
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)
        self.frames = {}
        for F in (StartPage, PageOne, PageTwo, PageThree, PageFour):
            page_name = F.__name__
            frame = F(parent=container, controller=self)
            self.frames[page_name] = frame
            frame.grid(row=0, column=0, sticky="nsew")
        self.show_frame("StartPage")

    def show_frame(self, page_name):
        frame = self.frames[page_name]
        frame.tkraise()

    def on_closing(self):
        if messagebox.askokcancel("Quit", "Are you sure?"):
            global names
            with open("nameslist.txt", "w") as f:
                for i in names:
                    f.write(i+" ")
            try:
                shutil.rmtree("dataset/test")
            except:
                pass
            self.destroy()


class StartPage(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller
        load = Image.open("backg.jpg")
        load = load.resize((150, 150), Image.ANTIALIAS)
        render = ImageTk.PhotoImage(load)
        img = tk.Label(self, image=render)
        img.image = render
        img.grid(row=0, column=1, rowspan=4, sticky="nsew")
        label = tk.Label(self, text="Face Recognizer", font=controller.title_font, fg="darkblue")
        label.grid(row=0, sticky="ew")
        button1 = tk.Button(self, text="Add a new user", fg="darkblue", bg="lightblue",
                            command=lambda: controller.show_frame("PageOne"))
        button2 = tk.Button(self, text="Select existing user", fg="darkblue", bg="lightblue",
                            command=lambda: controller.show_frame("PageTwo"))
        button3 = tk.Button(self, text="Quit", fg="darkblue", bg="lightblue", command=self.on_closing)
        button1.grid(row=2, column=0, ipady=3, ipadx=7)
        button2.grid(row=1, column=0, ipady=3, ipadx=0)
        button3.grid(row=3, column=0, ipady=2, ipadx=34)

    def on_closing(self):
        if messagebox.askokcancel("Quit", "Are you sure?"):
            global names
            with open("nameslist.txt", "w") as f:
                for i in names:
                    f.write(i+" ")
            try:
                shutil.rmtree("dataset/test")
            except:
                pass
            self.controller.destroy()


class PageOne(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller
        tk.Label(self, text="Enter the name", fg="blue", font='Helvetica 12 bold').grid(row=0, column=0, pady=10, padx=5)
        self.user_name = tk.Entry(self, borderwidth=3, bg="lightgrey", font='Helvetica 11')
        self.user_name.grid(row=0, column=1, pady=10, padx=10)
        self.controller.CNNobj = CNN()
        self.buttoncanc = tk.Button(self, text="Cancel", fg="red", bg="lightblue", command=lambda: controller.show_frame("StartPage"))
        self.buttonext = tk.Button(self, text="Next", fg="green", bg="lightblue", command=self.start_training)
        self.buttoncanc.grid(row=1, column=0, pady=10, ipadx=5, ipady=4)
        self.buttonext.grid(row=1, column=1, pady=10, ipadx=5, ipady=4)

    def start_training(self):
        global names
        if self.user_name.get() == "None":
            messagebox.showerror("Error", "Name cannot be 'None'")
            return
        if self.user_name.get() in names:
            messagebox.showerror("Error", "User already exists!")
            return
        if len(self.user_name.get()) == 0:
            messagebox.showerror("Error", "Name cannot be empty!")
            return
        name = self.user_name.get()
        names.add(name)
        self.controller.active_name = name
        self.controller.Buildposobj.set_name(name)
        self.controller.CNNobj.set_name(name)
        self.controller.frames["PageTwo"].refresh_names()
        self.controller.show_frame("PageThree")


class PageTwo(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        global names
        self.controller = controller
        tk.Label(self, text="Select user", fg="blue", font='Helvetica 12 bold').grid(row=0, column=0, padx=10, pady=10)
        self.buttoncanc = tk.Button(self, text="Cancel", command=lambda: controller.show_frame("StartPage"), fg="red", bg="lightblue")
        self.menuvar = tk.StringVar(self)
        self.dropdown = tk.OptionMenu(self, self.menuvar, *names)
        self.dropdown.config(bg="lightgrey")
        self.dropdown["menu"].config(bg="lightgrey")
        self.buttonext = tk.Button(self, text="Next", command=self.nextfoo, fg="green", bg="lightblue")
        self.dropdown.grid(row=0, column=1, ipadx=8, padx=10, pady=10)
        self.buttoncanc.grid(row=1, ipadx=5, ipady=4, column=0, pady=10)
        self.buttonext.grid(row=1, ipadx=5, ipady=4, column=1, pady=10)

    def nextfoo(self):
        if self.menuvar.get() == "None":
            messagebox.showerror("ERROR", "Name cannot be 'None'")
            return
        self.controller.active_name = self.menuvar.get()
        self.controller.CNNobj = LoadCnn(self.controller.active_name)
        self.controller.show_frame("PageFour")

    def refresh_names(self):
        global names
        self.menuvar.set('')
        self.dropdown['menu'].delete(0, 'end')
        for name in names:
            self.dropdown['menu'].add_command(label=name, command=tk._setit(self.menuvar, name))

class PageThree(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller
        self.numimglabel = tk.Label(self, text="Number of images captured = 0", font='Helvetica 12 bold', fg="blue")
        self.numimglabel.grid(row=0, column=0, columnspan=2, sticky="ew", pady=10)
        self.capturebutton = tk.Button(self, text="Capture images", fg="darkblue", bg="lightblue", command=self.capimg)
        self.trainbutton = tk.Button(self, text="Start training the model", fg="darkblue", bg="lightblue",
                            command=self.trainmodel)
        self.capturebutton.grid(row=1, column=0, ipadx=5, ipady=4, padx=10, pady=20)
        self.trainbutton.grid(row=1, column=1, ipadx=5, ipady=4, padx=10, pady=20)

    def capimg(self):
        messagebox.showinfo("INSTRUCTIONS", "Now your web cam will be opened. Capture at least 200 images with varied facial expressions. Press k to capture an image. Press q or ESC to exit.")
        self.controller.Buildposobj.start_capture()
        x = self.controller.Buildposobj.num_of_images
        self.numimglabel.config(text=str("Number of images captured = "+str(x)))

    def trainmodel(self):
        if self.controller.Buildposobj.num_of_images < 300:
            messagebox.showerror("ERROR", "Capture at least 300 images!")
            return
        self.controller.createtest.run()
        self.controller.CNNobj.compile()
        self.controller.CNNobj.create_train_test()
        self.controller.CNNobj.fit_generate()
        self.controller.CNNobj.classifier.save(str(self.controller.CNNobj.pers_name+".h5"))
        messagebox.showinfo("SUCCESS", "The CNN has been successfully trained!")
        self.controller.show_frame("PageFour")

class PageFour(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller
        label = tk.Label(self, text="Face Recognition", font='Helvetica 16 bold')
        label.grid(row=0, sticky="ew")
        button1 = tk.Button(self, text="Open web cam for Face Recognition",
                            command=self.openwebcam, fg="darkblue", bg="lightblue")
        button2 = tk.Button(self, text="Go to Home", command=lambda:self.controller.show_frame("StartPage"), fg="red", bg="lightblue")
        button1.grid(row=1, sticky="ew", ipadx=5, ipady=4, padx=10, pady=10)
        button2.grid(row=2, sticky="ew", ipadx=5, ipady=4, padx=10, pady=10)
    def openwebcam(self):
        DetectFace(self.controller.active_name, self.controller.CNNobj)

if __name__ == "__main__":
    app = MainUI()
    app.mainloop()