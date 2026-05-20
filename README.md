# 🍎 Fructoxis Vision Engine

Fructoxis Vision is a multi-stage machine learning pipeline designed to assess the quality of fruits (Apples, Bananas, Guavas, and Oranges). It categorizes fruits into three distinct states: **Good**, **Bad**, or **Mixed**.

**Links:**
- 📦 **Original Dataset (Kaggle)**: [FruitNet Indian Fruits Dataset with Quality](https://www.kaggle.com/datasets/shashwatwork/fruitnet-indian-fruits-dataset-with-quality)
- 💻 **GitHub Repository**: [iamdevkrishna/Fructoxis](https://github.com/iamdevkrishna/Fructoxis)

The engine leverages a dual-model approach:
1. **YOLOv8** (Spatial Detection): Identifies and crops the fruit from the raw image, ignoring backgrounds.
2. **MobileNetV2 CNN** (Biochemical Estimation): A fine-tuned model that processes the cropped fruit to classify its quality state.

---

## 📂 Project Structure

```text
Fructoxis/
│
├── data/                       # Datasets
│   ├── Fructoxis_Dataset/      # 12-Class organized dataset (Tracked, approx 200MB)
│   └── Processed Images_Fruits/# Pre-processed dataset variations (Not tracked)
│
├── src/
│   ├── app.py                  # Main Streamlit application frontend and pipeline logic
│   ├── prep_data.py            # Script for data preparation and augmentation
│   ├── train_model.py          # Script to fine-tune the MobileNetV2 architecture
│   ├── fructoxis_v2.keras      # (Not tracked) Fine-tuned CNN model weights
│   └── yolov8l.pt              # (Not tracked) YOLOv8 Large pre-trained weights
│
├── .gitignore                  # Git ignore rules
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation
```

---

## 🚀 Getting Started

### 1. Prerequisites
- Python 3.8+ installed on your system.
- Git.

### 2. Installation
Clone the repository and set up a virtual environment:

```bash
# Clone the repo
git clone https://github.com/iamdevkrishna/Fructoxis.git
cd Fructoxis

# Create a virtual environment
python -m venv .venv

# Activate the virtual environment
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Obtain the Models
Due to GitHub file size limits, model weights are strictly ignored in `.gitignore`. 
- **Fructoxis CNN (`fructoxis_v2.keras`)**: You can generate this by placing your training dataset in `data/Fructoxis_Dataset` and running `python src/train_model.py`.
- **YOLOv8 Weights (`yolov8l.pt`)**: Download [yolov8l.pt](https://github.com/ultralytics/assets/releases) and place it in the `src/` directory.

---

## 🏃‍♂️ How to Run

Once your models and environment are ready, you can launch the Vision Engine via Streamlit:

```bash
# Ensure you are at the root of the project (where this README is located)
cd src

# Run the Streamlit app
streamlit run app.py
```

Upload a clear image of an Apple, Banana, Guava, or Orange (JPG/PNG) to the target zone in the web interface to view the visual analysis and status!

---

## 🧠 Architecture Overview
- **Stage 1 (YOLO Target Isolation)**: Scans the raw feed, filters out irrelevant objects (like backgrounds or tables), and yields a strictly bounded box of the primary object. 
- **Stage 2 (CNN Assessment)**: Applies image normalization matching the `MobileNetV2` preprocess state and computes a probability distribution against 12 underlying fruit & condition classes.
