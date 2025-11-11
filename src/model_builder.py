import tensorflow as tf
import os
import yaml

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
PARAMS_PATH = os.path.join(PROJECT_ROOT, "params.yaml")

def build_model(img_size):
    
    
    IMG_SIZE_TUPLE = (img_size, img_size)

    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(IMG_SIZE_TUPLE[0], IMG_SIZE_TUPLE[1], 3),
        include_top=False,  
        weights='imagenet' 
    )

    base_model.trainable = False

    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(1, activation='sigmoid') 
    ])

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=[
            'accuracy',
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall')
        ]
    )
    
    print("Model built and compiled successfully.")
    return model

if __name__ == "__main__":
    
    try:
        with open(PARAMS_PATH, 'r') as f:
            params = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: '{PARAMS_PATH}' not found.")
        print("Please ensure 'params.yaml' is in your project's root directory.")
        exit()
    except Exception as e:
        print(f"Error loading {PARAMS_PATH}: {e}")
        exit()

    model_params = params['model']
    
    model = build_model(img_size=model_params['img_size'])
    
    print("\n--- model_builder.py Test Run ---")
    model.summary()
    print("\nModel builder script test successful.")