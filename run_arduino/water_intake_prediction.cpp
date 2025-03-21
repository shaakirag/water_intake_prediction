#include <TensorFlowLite.h>
#include "calorie_nn_quant.h"  // Include the model as a C array

// Set up the interpreter
tflite::MicroErrorReporter micro_error_reporter;
tflite::ErrorReporter* error_reporter = &micro_error_reporter;

// Load the model
const tflite::Model* model = tflite::GetModel(calorie_nn_quant);
if (model->version() != TFLITE_SCHEMA_VERSION) {
  error_reporter->Report("Model version does not match Schema version!");
  while (1);
}

// Set up the interpreter
constexpr int kTensorArenaSize = 16 * 1024;  // Adjust based on model size
uint8_t tensor_arena[kTensorArenaSize];
tflite::MicroInterpreter interpreter(model, tensor_arena, kTensorArenaSize, error_reporter);

// Allocate tensors
interpreter.AllocateTensors();

// Get input and output tensors
TfLiteTensor* input = interpreter.input(0);
TfLiteTensor* output = interpreter.output(0);

// Prepare input data ([Gender, Age, Weight, Duration, Heart_Rate]) See README for data units
float input_data[5] = {1.0, 25.0, 70.0, 100.0, 100.0};  // Replace with actual input
for (int i = 0; i < 5; i++) {
  input->data.int8[i] = (int8_t)(input_data[i] / input->params.scale + input->params.zero_point);
}

// Run inference
if (interpreter.Invoke() != kTfLiteOk) {
  error_reporter->Report("Inference failed!");
  while (1);
}

// Get the output
float predicted_calories = output->data.f[0];
float water_required = predicted_calories / 2.42;  // Calculate water required

// Print results
Serial.print("Predicted Calories: ");
Serial.println(predicted_calories);
Serial.print("Water Required: ");
Serial.println(water_required);