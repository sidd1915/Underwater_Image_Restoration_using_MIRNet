import gradio as gr
from PIL import Image
import numpy as np
from UDCP.udcp import apply_udcp  # UDCP method
# from UDCP.RefinedTramsmission import Refinedtransmission
# from UDCP.getAtomsphericLight import getAtomsphericLight
# from UDCP.getGbDarkChannel import getDarkChannel
# from UDCP.getTM import getTransmission
# from UDCP.sceneRadiance import sceneRadianceRGB
from CLAHE.clahe import apply_clahe
from WB.wb import gray_world
from MIRNet.mirnet import restore_image


restored_image_cache = None

def restore_image_with_mirnet(image):
    global restored_image_cache
    restored_image_cache = restore_image(image)
    return restored_image_cache, "Restoration Complete"

def enhance_image(enhancement_option):
    global restored_image_cache
    if restored_image_cache is None:
        return None, "Please restore the image first."

    if enhancement_option == "UDCP":
        enhanced_image = apply_udcp(restored_image_cache)
    elif enhancement_option == "CLAHE":
        enhanced_image = apply_clahe(restored_image_cache)
    elif enhancement_option == "White Balance":
        restored_image_cache = Image.fromarray(restored_image_cache)
        enhanced_image = gray_world(restored_image_cache)
        restored_image_cache = np.array(restored_image_cache)
    else:
        enhanced_image = restored_image_cache  # No enhancement selected

    return enhanced_image, f"{enhancement_option} Enhancement Applied"


# Gradio functions
def process_image(image):
    # Restore image using MIRNet
    restored_image = restore_image_with_mirnet(image)
    return restored_image, None  # Initial enhancement is None

def process_enhancement(enhancement_option):
    enhanced_image = enhance_image(enhancement_option)
    return enhanced_image


# Custom CSS to change background, text color, and other UI elements, supporting both light and dark modes
css = """
/* Default light mode styles */
body {
    background-color: white;  /* Set the overall background to white */
    font-family: Arial, sans-serif;
    color: black;  /* Set default text color to black */
}

.gradio-container {
    background-color: #f0f8ff;  /* Set container background to light blue */
    border-radius: 10px;
    padding: 20px;
    color: black;  /* Ensure text inside the container is black */
}

button {
    background-color: #4CAF50;  /* Green button */
    color: white;
    border: none;
    border-radius: 5px;
    padding: 10px 20px;
    font-size: 16px;
    cursor: pointer;
}

button:hover {
    background-color: #45a049;  /* Darker green when hovering */
}

.gradio-image-container {
    border-radius: 8px;
    border: 2px solid #ddd;  /* Gray border for images */
}

.gradio-textbox {
    color: black;  /* Set text color to black in textboxes */
}

.gradio-markdown {
    color: black;  /* Set text color to black in markdown */
}

/* Dark mode styles */
@media (prefers-color-scheme: dark) {
    body {
        background-color: #121212;  /* Dark background for body */
        color: white;  /* White text */
    }

    .gradio-container {
        background-color: #333333;  /* Dark background for container */
        color: white;  /* White text inside container */
    }

    button {
        background-color: #4CAF50;  /* Green button */
        color: white;
    }

    .gradio-image-container {
        border-radius: 8px;
        border: 2px solid #444;  /* Dark border for images */
    }

    .gradio-textbox {
        color: white;  /* White text in textboxes */
    }

    .gradio-markdown {
        color: black;  /* White text in markdown */
    }
}
"""

# Gradio interface with updated CSS
import gradio as gr

# Define the app functionality and layout here
# (Keep the same code structure as shown in previous examples)

theme = gr.themes.Ocean()

with gr.Blocks(theme=theme) as interface:
    gr.Markdown("# 🌊 **Underwater Image Restoration and Enhancement App**")
    gr.Markdown(
        "This application restores underwater images using **MIRNet** and offers optional enhancement using "
        "algorithms like **UDCP**, **CLAHE**, or **White Balance**."
    )
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("### Upload Image")
            input_image = gr.Image(type="numpy", label="Upload Underwater Image")
            restore_button = gr.Button("Restore Image")
        with gr.Column():
            gr.Markdown("### Restored Image")
            restored_output = gr.Image(type="numpy", label="Restored Image (MIRNet)", interactive=False)
            status = gr.Textbox(label="Status", interactive=False, max_lines=1)

    gr.Markdown("### Enhancement Options")
    with gr.Tabs():
        with gr.TabItem("Enhance Image"):
            enhancement_options = gr.Radio(
                choices=["None", "UDCP", "CLAHE", "White Balance"],
                label="Choose Enhancement",
                value="None"
            )
            enhance_button = gr.Button("Apply Enhancement")
            enhanced_output = gr.Image(type="numpy", label="Enhanced Image", interactive=False)

    # Event Handling (Restoration and Enhancement Logic)
    restore_button.click(
        fn=restore_image_with_mirnet,
        inputs=input_image,
        outputs=[restored_output, status]
    )

    enhance_button.click(
        fn=enhance_image,
        inputs=enhancement_options,
        outputs=[enhanced_output, status]
    )

# Launch the app
if __name__ == "__main__":
    interface.launch()

