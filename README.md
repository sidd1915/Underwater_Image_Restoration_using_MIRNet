# Underwater Image Restoration using Modified MIRNet Architecture

This repository contains the implementation of underwater image restoration using a modified version of the MIRNet architecture. Our model is trained on the **UIEB dataset** to enhance underwater images by improving visibility, restoring natural colors, and reducing distortion caused by light absorption and scattering. The modifications to the original MIRNet architecture are designed to better preserve color and texture in underwater scenarios.

---

## Architecture Overview

Below is a visual representation of our modified MIRNet architecture:

![Modified MIRNet Architecture](./assets/architecture.png)

---

## Results

Here are some examples of underwater image restoration results achieved by our model:

| Input Image | Restored Image |
|-------------|----------------|
| ![Input Image](./assets/input_image.png) | ![Restored Image](./assets/restored_image.png) |

| Input Image | Restored Image |
|-------------|----------------|
| ![Input Image 2](./assets/input_image2.png) | ![Restored Image 2](./assets/restored_image2.png) |

---

## Usage

Follow the steps below to use the underwater image restoration application:

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/underwater-image-restoration.git
   cd underwater-image-restoration
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the Gradio application:
   ```bash
   python gradio_app.py
   ```

4. Access the application in your browser at the URL provided in the terminal (usually `http://127.0.0.1:7860/`).

---

---

## Citation

If you find this work useful, please cite the original MIRNet paper:

```bibtex
@inproceedings{zamir2020mirnet,
  title={Learning Enriched Features for Real Image Restoration and Enhancement},
  author={Syed Waqas Zamir and Aditya Arora and Salman Khan and Munawar Hayat and Fahad Shahbaz Khan and Ming-Hsuan Yang},
  booktitle={ECCV},
  year={2020}
}
```

---

## License

This project is licensed under the MIT License. See the [LICENSE](./LICENSE) file for more details.

---

## Acknowledgements

We extend our gratitude to the authors of MIRNet and the UIEB dataset for their contributions to the field of underwater image restoration.
