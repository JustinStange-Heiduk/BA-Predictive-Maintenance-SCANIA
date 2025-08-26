import tensorflow as tf

print("🚀 TensorFlow Setup gestartet.")
print(f"TensorFlow-Version: {tf.__version__}")

# Verfügbare GPUs anzeigen
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"✅ GPU(s) erkannt: {[gpu.name for gpu in gpus]}")
else:
    print("⚠️ Keine GPU erkannt – bitte CUDA/Treiber/Docker prüfen.")
