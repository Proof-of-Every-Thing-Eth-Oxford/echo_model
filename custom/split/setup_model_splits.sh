#!/bin/zsh

# Enable strict mode: Stop execution if any command fails
set -e

# Loop from 0 to 4
for i in {3..4}; do
    model_name="temp_sub_model_${i}.onnx"
    data_name="input_data_${i}.json"
    settings_name="settings_${i}.json"
    compiled_circuit="compiled_circuit_${i}"
    vk_path="vk_${i}.vk"
    pk_path="pk_${i}.pk"

    echo "▶️ Processing model: $model_name with data: $data_name"

    # Step 1: Generate settings
    echo "🔹 Running: gen-settings --model "$model_name" --settings-path "$settings_name""
    ezkl gen-settings --model "$model_name" --settings-path "$settings_name" || { echo "❌ Error: gen-settings failed for $model_name"; exit 1; }

    # Step 2: Calibrate settings
    echo "🔹 Running: ezkl calibrate-settings --model $model_name --data $data_name --settings-path $settings_name --target performance --max-logrows 20"
    ezkl calibrate-settings --model "$model_name" --data "$data_name" --settings-path "$settings_name" || { echo "❌ Error: calibrate-settings failed for $model_name"; exit 1; }

    # Step 3: Compile the circuit
    echo "🔹 Running: ezkl compile-circuit --model "$model_name" --compiled-circuit "$compiled_circuit""
    ezkl compile-circuit --model "$model_name" --compiled-circuit "$compiled_circuit" || { echo "❌ Error: compile-circuit failed for $model_name"; exit 1; }

    # Step 4: Generate PK and VK
    echo "🔹 Running: ezkl setup --vk-path "$vk_path" --pk-path "$pk_path" --compiled-circuit "$compiled_circuit""
    ezkl setup --vk-path "$vk_path" --pk-path "$pk_path" --compiled-circuit "$compiled_circuit" || { echo "❌ Error: setup failed for $model_name"; exit 1; }

    echo "✅ Completed processing for $model_name"
done

echo "🎉 All models processed successfully!"