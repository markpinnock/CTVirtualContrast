import tkinter as tk


window = tk.Tk()
window.geometry("650x250+1000+600")
print(window.winfo_screenwidth(), window.winfo_screenheight())
window.mainloop()