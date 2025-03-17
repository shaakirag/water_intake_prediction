# Calorie Prediction Model

This project provides a water intake prediction model that can be run in both Python and Arduino environments. The model predicts calories burned based on input features such as gender, age, weight, duration of activity, and heart rate.

---

## Data Requirements

The model requires the following input features:

- **Gender**: 
  - `1` for Male
  - `0` for Female
- **Age**: 
  - In years (e.g., `25`)
- **Weight**: 
  - In kilograms (e.g., `70`)
- **Duration**: 
  - In minutes (e.g., `30`)
- **Heart Rate**: 
  - In beats per minute (BPM) (e.g., `100`)

---

## Model Training Results

For detailed results of the model training, refer to the [Google Slides presentation](https://docs.google.com/presentation/d/1pbetKu93cnkbk1hja1JKAw4dXaC1zY9MZkWzAQAawM0/edit?usp=sharing).

---

## Running the Model

### In a Python Environment

1. **Navigate to the `run_python` directory**:
   ```bash
   cd run_python
   ```
2. **Download the Required Files**:
    - Download the HDF5 model file (`calorie_nn.h5`).
    - Download the Jupyter Notebook file (`water_intake_prediction.ipynb`).
    - Make sure both files are in the same directory.
4. **Run the First Cell**:
    - Open the Jupyter Notebook and:
      - Provide your input data as specified in the **Data Requirements** section.
      - Alternatively, test the model using the dataset provided in `misc/dataset`.
    - Run the first cell.
5. **Ignore Other Cells**:
    - The remaining cells are used to create a lightweight model for the Arduino and can be ignored unless you need to retrain or convert the model.

---

### On an Arduino

1. **Navigate to the `run_arduino` Directory**:
   ```bash
   cd run_arduino
   ```
2. **Download the Required Files**:
    - Download the HDF5 model file (`calorie_nn_quant.h`).
    - Download the Jupyter Notebook file (`water_intake_prediction.cpp`).
3. **Upload and Run the code**:
    - Open the Arduino IDE.
    - Upload the starter code and header file to your Arduino board.
    - Provide your input data as specified in the Data Requirements section.
    - Alternatively, test the model using sample inputs from the dataset in `misc/dataset`.
       
## Notes

### Header File for Arduino
  - The header file (`calorie_nn_quant.h`) in the `run_arduino` directory is a compressed, quantized version of the model designed for microcontrollers. It is lightweight and optimized for Arduino.

### TensorFlow Lite Models
  - TensorFlow Lite models can be found in the `misc/models` directory.

### Training Code
  - The code used to train the model is available in the `misc/training` directory.
