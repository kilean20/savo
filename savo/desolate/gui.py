import tkinter as tk
import numpy as np

class popup_handler:
    """
    A class for creating a popup window with dynamic sizing based on message length.

    Attributes:
        title (str): The title of the popup window.
        message (str): The message to be displayed in the popup window.
        root (tk.Tk): The Tkinter root window.
        blink_interval (int): The interval for blinking background in milliseconds.

    Usage:
        popup = PopupHandler("Title", "Your long message goes here.")
        popup()
    """

    def __init__(self, title: str, message: str, blink_interval: int = 500):
        """
        Initializes a PopupHandler instance.

        Args:
            title (str): The title of the popup window.
            message (str): The message to be displayed in the popup window.
            blink_interval (int): The interval for blinking background in milliseconds.
        """
        self.message = self.format_message(message)
        self.title = title
        self.root = None
        self.blink_interval = blink_interval
        self.blink_background = False  # Flag to track blinking state

    def format_message(self, message: str) -> str:
        """
        Formats the message by inserting newlines after around 50 characters without breaking words.

        Args:
            message (str): The original message.

        Returns:
            str: The formatted message.
        """
        words = message.split()
        lines = []
        current_line = ""
        for word in words:
            if len(current_line) + len(word) <= 50:
                current_line += word + " "
            else:
                lines.append(current_line.strip())
                current_line = word + " "
        if current_line:
            lines.append(current_line.strip())
        return '\n'.join(lines)

    def on_close(self):
        """
        Handles the window close event (e.g., when the user clicks the 'X' button).
        For simplicity, continues the program when the window is closed.
        """
        self.root.destroy()

    def toggle_blink(self):
        """Toggle blinking background."""
        if self.blink_background:
            self.root.configure(background='white')
        else:
            self.root.configure(background='yellow')
        self.blink_background = not self.blink_background
        self.root.after(self.blink_interval, self.toggle_blink)

    def __call__(self, message=None, button_txt=None):
        """
        Displays the popup window.
        """
        if message is None:
            message = self.message
        if button_txt is None:
            button_txt = 'Continue'
        self.root = tk.Tk()
        self.root.title(self.title)

        # Adjust window size based on the message length and button height
        max_line_length = len(max(self.message.split('\n'), key=len))
        num_lines = len(self.message.split('\n'))
        button_height = 40
        width = min(400, max_line_length * 10)
        height = np.clip(num_lines * 20 + 2*button_height, a_min= 20 + 2*button_height, a_max=860)
        self.root.geometry(f"{width}x{height}")

        self.root.attributes('-topmost', True)  # Make the window appear on top

        # Add small padding at the top for the label
        self.label = tk.Label(self.root, text=message, padx=10, pady=10, justify='left', wraplength=width - 20)
        self.label.pack(pady=(10, 0))

        # Move the button to the bottom
        self.button = tk.Button(self.root, text=button_txt, command=self.on_close)
        self.button.pack(side=tk.BOTTOM, pady=10, anchor='s')

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)  # handle window close event

        # Start blinking background
        self.root.after(0, self.toggle_blink)

        self.root.deiconify()  # show the window
        self.root.mainloop()
