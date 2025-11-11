import tensorflow as tf
import os
import yaml

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
PARAMS_PATH = os.path.join(PROJECT_ROOT, "params.yaml")

def get_data_generators(train_dir, test_dir, img_size, batch_size, val_split, seed):
 
    IMG_SIZE_TUPLE = (img_size, img_size)
    
    train_dir_abs = os.path.join(PROJECT_ROOT, train_dir)
    test_dir_abs = os.path.join(PROJECT_ROOT, test_dir)
    
    print(f"Loading training data from: {train_dir_abs}")
    print(f"Loading test data from: {test_dir_abs}")
    
    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir_abs, 
        label_mode='binary',
        image_size=IMG_SIZE_TUPLE,
        batch_size=batch_size,
        shuffle=True,
        validation_split=val_split,
        subset='training',
        seed=seed
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir_abs,
        label_mode='binary',
        image_size=IMG_SIZE_TUPLE,
        batch_size=batch_size,
        shuffle=False,
        validation_split=val_split,
        subset='validation',
        seed=seed
    )

    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_dir_abs,
        label_mode='binary',
        image_size=IMG_SIZE_TUPLE,
        batch_size=batch_size,
        shuffle=False
    )
    
    class_names = train_ds.class_names
    print(f"Classes found: {class_names}")
    
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomZoom(0.1),
        tf.keras.layers.Rescaling(1./255) 
    ])

    
    train_ds = train_ds.map(lambda x, y: (data_augmentation(x, training=True), y), 
                            num_parallel_calls=tf.data.AUTOTUNE)
    train_ds = train_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

    rescale_layer = tf.keras.layers.Rescaling(1./255)
    
    val_ds = val_ds.map(lambda x, y: (rescale_layer(x), y), 
                        num_parallel_calls=tf.data.AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

    test_ds = test_ds.map(lambda x, y: (rescale_layer(x), y), 
                          num_parallel_calls=tf.data.AUTOTUNE)
    test_ds = test_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
    
    print("Data augmentation and rescaling applied.")
    
    return train_ds, val_ds, test_ds, class_names

if __name__ == "__main__":
    # This block is for testing the script directly
    
    # Load parameters
    try:
        # We now use the robust PARAMS_PATH
        with open(PARAMS_PATH, 'r') as f:
            params = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: '{PARAMS_PATH}' not found.")
        print("Please ensure 'params.yaml' is in your project's root directory.")
        exit()
    except Exception as e:
        print(f"Error loading {PARAMS_PATH}: {e}")
        exit()

    data_params = params['data']
    model_params = params['model']
    
    # Get data
    train_ds, val_ds, test_ds, classes = get_data_generators(
        train_dir=data_params['train_dir'],
        test_dir=data_params['test_dir'],
        img_size=model_params['img_size'],
        batch_size=model_params['batch_size'],
        val_split=data_params['val_split'],
        seed=data_params['seed']
    )
    
    print("\n--- data_loader.py Test Run ---")
    print(f"train_ds: {train_ds}")
    print(f"val_ds: {val_ds}")
    print(f"test_ds: {test_ds}")
    print(f"Classes: {classes}")
    print("Data loading script test successful.")