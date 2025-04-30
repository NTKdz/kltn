import win32gui
import win32ui
import win32con
import win32api
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import ctypes
from ctypes import wintypes
import numpy as np

# Define Windows API types
user32 = ctypes.WinDLL('user32', use_last_error=True)
gdi32 = ctypes.WinDLL('gdi32', use_last_error=True)
psapi = ctypes.WinDLL('psapi', use_last_error=True)  # Load psapi.dll directly

# Structure to hold window information
class WindowInfo:
    def __init__(self, hwnd, title, process_name):
        self.hwnd = hwnd
        self.title = title
        self.process_name = process_name

    def __str__(self):
        return f"{self.process_name} - {self.title}"

# List to store visible windows
windows = []

# Callback function for EnumWindows
def enum_windows_proc(hwnd, lparam):
    try:
        if user32.IsWindowVisible(hwnd):
            title_length = user32.GetWindowTextLengthW(hwnd) + 1
            title_buffer = ctypes.create_unicode_buffer(title_length)
            user32.GetWindowTextW(hwnd, title_buffer, title_length)
            title = title_buffer.value.strip()
            if title:
                # Get process ID
                pid = wintypes.DWORD()
                user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
                # Open process to get process name
                h_process = ctypes.windll.kernel32.OpenProcess(
                    win32con.PROCESS_QUERY_INFORMATION | win32con.PROCESS_VM_READ, False, pid.value
                )
                if h_process:
                    process_name_buffer = ctypes.create_unicode_buffer(512)
                    psapi.GetModuleBaseNameW(h_process, None, process_name_buffer, 512)
                    process_name = process_name_buffer.value
                    ctypes.windll.kernel32.CloseHandle(h_process)
                    windows.append(WindowInfo(hwnd, title, process_name))
    except Exception as e:
        print(f"Error enumerating window {hwnd}: {e}")
    return True

# Function to capture a window's content
def capture_window(hwnd):
    try:
        # Get window rectangle
        rect = wintypes.RECT()
        if not user32.GetWindowRect(hwnd, ctypes.byref(rect)):
            print("Failed to get window rect")
            return None

        width = rect.right - rect.left
        height = rect.bottom - rect.top

        # Get window DC
        hwnd_dc = user32.GetDC(hwnd)
        mem_dc = gdi32.CreateCompatibleDC(hwnd_dc)
        bitmap = gdi32.CreateCompatibleBitmap(hwnd_dc, width, height)
        old_bitmap = gdi32.SelectObject(mem_dc, bitmap)

        # Try PrintWindow first
        success = user32.PrintWindow(hwnd, mem_dc, 0)
        if not success:
            # Fallback to BitBlt
            success = gdi32.BitBlt(mem_dc, 0, 0, width, height, hwnd_dc, 0, 0, win32con.SRCCOPY)

        if success:
            # Create a BITMAPINFO structure for GetDIBits
            class BITMAPINFOHEADER(ctypes.Structure):
                _fields_ = [
                    ('biSize', wintypes.DWORD),
                    ('biWidth', wintypes.LONG),
                    ('biHeight', wintypes.LONG),
                    ('biPlanes', wintypes.WORD),
                    ('biBitCount', wintypes.WORD),
                    ('biCompression', wintypes.DWORD),
                    ('biSizeImage', wintypes.DWORD),
                    ('biXPelsPerMeter', wintypes.LONG),
                    ('biYPelsPerMeter', wintypes.LONG),
                    ('biClrUsed', wintypes.DWORD),
                    ('biClrImportant', wintypes.DWORD),
                ]

            class BITMAPINFO(ctypes.Structure):
                _fields_ = [
                    ('bmiHeader', BITMAPINFOHEADER),
                    ('bmiColors', wintypes.DWORD * 1),
                ]

            bmi = BITMAPINFO()
            bmi.bmiHeader.biSize = ctypes.sizeof(BITMAPINFOHEADER)
            bmi.bmiHeader.biWidth = width
            bmi.bmiHeader.biHeight = -height  # Top-down
            bmi.bmiHeader.biPlanes = 1
            bmi.bmiHeader.biBitCount = 32
            bmi.bmiHeader.biCompression = win32con.BI_RGB

            # Create buffer for pixel data
            pixel_data = np.zeros((height, width, 4), dtype=np.uint8)
            gdi32.GetDIBits(
                mem_dc, bitmap, 0, height, pixel_data.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
                ctypes.byref(bmi), win32con.DIB_RGB_COLORS
            )

            # Convert to PIL Image
            image = Image.fromarray(pixel_data, 'RGBA')
        else:
            image = None

        # Cleanup
        gdi32.SelectObject(mem_dc, old_bitmap)
        gdi32.DeleteObject(bitmap)
        gdi32.DeleteDC(mem_dc)
        user32.ReleaseDC(hwnd, hwnd_dc)

        return image
    except Exception as e:
        print(f"Capture failed: {e}")
        return None

# GUI Application
class ScreenPickerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Screen Picker")
        self.root.geometry("600x400")

        # Top panel
        top_frame = tk.Frame(self.root)
        top_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)

        tk.Label(top_frame, text="Select Window:").pack(side=tk.LEFT)
        self.window_combobox = ttk.Combobox(top_frame, values=[str(w) for w in windows], state="readonly")
        self.window_combobox.pack(side=tk.LEFT, padx=5)
        tk.Button(top_frame, text="Capture Window", command=self.capture).pack(side=tk.LEFT)

        # Canvas for image display
        self.canvas = tk.Canvas(self.root)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        self.image = None
        self.photo = None

    def capture(self):
        if self.window_combobox.current() == -1:
            tk.messagebox.showerror("Error", "No window selected.")
            return
        selected_window = windows[self.window_combobox.current()]
        image = capture_window(selected_window.hwnd)
        if image:
            # Resize image to fit canvas if needed
            canvas_width = self.canvas.winfo_width()
            canvas_height = self.canvas.winfo_height()
            if canvas_width > 0 and canvas_height > 0:
                image = image.resize((canvas_width, canvas_height), Image.Resampling.LANCZOS)

            self.image = image
            self.photo = ImageTk.PhotoImage(self.image)
            self.canvas.delete("all")
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)
        else:
            tk.messagebox.showerror("Error", "Failed to capture window.")

# Main execution
if __name__ == "__main__":
    # Enumerate windows
    enum_windows_proc_type = ctypes.WINFUNCTYPE(ctypes.c_bool, wintypes.HWND, wintypes.LPARAM)
    user32.EnumWindows(enum_windows_proc_type(enum_windows_proc), 0)

    # Create GUI
    root = tk.Tk()
    app = ScreenPickerApp(root)
    root.mainloop()