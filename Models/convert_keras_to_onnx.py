import os
import tensorflow as tf
import tf2onnx

# Path where your .keras models are stored
MODEL_DIR = "."

# Loop through all .keras files
keras_files = [f for f in os.listdir(MODEL_DIR) if f.endswith(".keras")]

if not keras_files:
    print("⚠️ No .keras files found in the directory.")
else:
    print(f"Found {len(keras_files)} Keras models:")
    for f in keras_files:
        print(f" - {f}")

    for keras_file in keras_files:
        keras_path = os.path.join(MODEL_DIR, keras_file)
        onnx_path = keras_path.replace(".keras", ".onnx")

        print(f"\n🔄 Converting: {keras_file}")

        try:
            # Load model without compiling (ignore loss and optimizer)
            model = tf.keras.models.load_model(keras_path, compile=False)

            # Convert to ONNX
            spec = (tf.TensorSpec(model.inputs[0].shape, tf.float32, name="input"),)
            model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec, output_path=onnx_path)

            print(f"✅ Saved: {onnx_path}")
        except Exception as e:
            print(f"❌ Failed to convert {keras_file}: {e}")

    print("\n🎉 Conversion attempt complete!")
