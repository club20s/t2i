import tkinter as tk
from tkinter import PhotoImage
from PIL import Image, ImageTk
import torch
from torch.cuda.amp import autocast
from diffusers import StableDiffusionPipeline
from datetime import datetime
from transformers import CLIPImageProcessor
import os
import matplotlib.pyplot as plt

os.environ['TK_SILENCE_DEPRECATION'] = '1'

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.default_window_width = 1200
        self.default_window_height = 800
        self.authorization_token = "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"

        self.title("Image Generator")
        self.geometry(f"{self.default_window_width}x{self.default_window_height}")
        self.configure(background='black')

        self.windowlabel = tk.Label(self, text="AI ART Image Generator", font=("Arial", 30, "bold"), padx=50, pady=50, fg="white", bg="black")
        self.windowlabel.pack()

        self.promptlabel = tk.Label(self, text="Prompt", font=("Montserrat", 20, "bold"), fg="white", bg="black")
        self.promptlabel.pack()

        self.promptentry = tk.Entry(self, width=self.default_window_width-20, font=("Arial", 12), bd=2)
        self.promptentry.pack(padx=20, pady=20)

        self.guidancelabel = tk.Label(self, text="Guidance Scale", font=("Montserrat", 20, "bold"), fg="white", bg="black")
        self.guidancelabel.pack()

        self.guidanceentry = tk.Entry(self, width=10, font=("Arial", 12), bd=2)
        self.guidanceentry.pack(padx=20, pady=20)

        self.generatebutton = tk.Button(self, text="Generate Image", width=self.default_window_width-50, height=2, fg="black", bd=2, bg="white", command=self.generate)
        self.generatebutton.pack()

    def generate(self):
        self.textprompt = self.promptentry.get()
        self.guidance_scale = float(self.guidanceentry.get())
        self.generatebutton.configure(state="disabled")

        self.progress = tk.Label(self, text="Generating Image...", font=("Arial", 12), fg="white", bg="black")
        self.progress.pack()

        modelid = "runwayml/stable-diffusion-v1-5"
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        try:
            # Load pipeline and image processor
            image_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32")
            pipe = StableDiffusionPipeline.from_pretrained(modelid, variant="fp16", token=self.authorization_token)
            pipe.to(device)

            # Timestamp saat proses dimulai
            start_time = datetime.now()

            with autocast(device):
                result = pipe(prompt=self.textprompt, guidance_scale=self.guidance_scale)
                image = result.images[0]

                # Display image with matplotlib
                plt.imshow(image)
                plt.title(f"Guidance Scale: {self.guidance_scale}")
                plt.show()

                # Save image with guidance scale in filename
                image_filename = f'generated_image_guidance_{self.guidance_scale}.png'
                image.save(image_filename)
                img = ImageTk.PhotoImage(image)

                self.imageview = tk.Label(self, image=img)
                self.imageview.image = img
                self.imageview.pack()

                self.alt_text = tk.Label(self, text="Generated Image", font=("Arial", 12), fg="white", bg="black")
                self.alt_text.pack()

            # Timestamp saat proses selesai
            end_time = datetime.now()

            # Hitung durasi proses
            duration = end_time - start_time
            print(f"Image generation took: {duration}")

        except Exception as e:
            print(f"Error during image generation: {str(e)}")

        self.progress.pack_forget()
        self.generatebutton.configure(state="normal")

if __name__ == "__main__":
    app = App()
    app.mainloop()
