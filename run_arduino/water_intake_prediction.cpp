#include <TensorFlowLite.h>
#include "calorie_nn_quant.h"  // Model as a C array

tflite::MicroErrorReporter micro_error_reporter;
tflite::ErrorReporter* error_reporter = &micro_error_reporter;

const tflite::Model* model = tflite::GetModel(calorie_nn_quant);
if (model->version() != TFLITE_SCHEMA_VERSION) {
  error_reporter->Report("Model version mismatch!");
  while (1);
}

// Increase arena size to 20KB for safety
constexpr int kTensorArenaSize = 20 * 1024;
uint8_t tensor_arena[kTensorArenaSize];

tflite::MicroInterpreter interpreter(model, tensor_arena, kTensorArenaSize, error_reporter);
interpreter.AllocateTensors();

TfLiteTensor* input = interpreter.input(0);
TfLiteTensor* output = interpreter.output(0);

// Data: [Gender, Age, Weight, Duration, Heart_Rate] See README for units
float input_data[5] = {1.0, 25.0, 70.0, 100.0, 100.0};

// Quantize input
for (int i = 0; i < 5; i++) {
  float scaled_value = input_data[i] / input->params.scale + input->params.zero_point;
  input->data.int8[i] = static_cast<int8_t>(round(scaled_value));
}

// Run inference
if (interpreter.Invoke() != kTfLiteOk) {
  error_reporter->Report("Inference failed!");
  while (1);
}

// Dequantize output
int8_t output_quant = output->data.int8[0];
float predicted_calories = (output_quant - output->params.zero_point) * output->params.scale;

float water_required = predicted_calories / 2.42;

Serial.print("Predicted Calories: ");
Serial.println(predicted_calories);
Serial.print("Water Required: ");
Serial.println(water_required);