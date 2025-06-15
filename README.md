
# PhokshoSathi 🫁

**PhokshoSathi** (translates to *Lung Companion*) is a Python-based desktop application powered by a fine-tuned **ResNet50** model that classifies chest X-ray scans as **Pneumonia Positive** or **Negative**. Designed with a user-friendly GUI, it's ideal for rapid screening in clinical or educational settings.

---

## 🧠 Model Overview

- **Architecture:** Pretrained [ResNet50](https://arxiv.org/abs/1512.03385)
- **Modification:** Final fully connected layer fine-tuned for binary classification of chest X-rays
- **Training Accuracy:** ~91%

---

## 💡 Features

- ✅ High-accuracy pneumonia detection
- 🖥️ Fullscreen, desktop-friendly GUI
- 🐍 Built entirely with Python (PyQt5)
- ⚡ Fast, lightweight, and easy to run locally

---



## Installation

Clone the project

```bash
git clone https://github.com/manishdhakal2/PhokshoSathi.git
```

Navigate to application directory

```bash
cd path/PhokshoSathi/GUI
```

Run the App

```bash
python app.py
```


    
## Roadmap

- Browser-Based GUI Support

- Add multi-class classification for other chest diseases

- Fine tuning for better accuracy


## Support or Feedback

For support or feedback, email dhakalmanish398@gmail.com


## License

[MIT](https://choosealicense.com/licenses/mit/)

