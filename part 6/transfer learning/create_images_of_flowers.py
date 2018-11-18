import glob
import os.path
import tensorflow as tf
from tensorflow.python.platform import gfile
import numpy as np
from tqdm import tqdm

FLOWER_PATH = "../../flower_photo/flower_photos"
OUTPUT_FILE = "./processed_flower_data.npy"
VALIDATION_PERCENTAGE = 10
TEST_PERCENTAGE = 10

def create_image(sess):
    
    train_image, train_label = [], []
    test_image, test_label = [], []
    validation_image, validation_label = [], []
    
    sub_dirs = [x[0] for x in os.walk(FLOWER_PATH)]
    is_root_path = True
    label_value = 0
    for sub_dir in tqdm(sub_dirs):
        if is_root_path:
            is_root_path = False
            continue
            
        dir_name = os.path.basename(sub_dir)
        extensions = ["jpg", "jpeg", "JPG", "JPEG"]
        image_list = []
        for extension in extensions:
            file_glob_pattern = os.path.join(FLOWER_PATH, dir_name, "*." + extension)
            image_list.extend(glob.glob(file_glob_pattern))
        if not image_list: continue

        for image in image_list:
            image_raw = gfile.FastGFile(image, "rb").read()
            image = tf.image.decode_jpeg(image_raw)
            if image.dtype != tf.float32:
                image = tf.image.convert_image_dtype(image, tf.float32)
            image = tf.image.resize_images(image, [299, 299])
            image_value = sess.run(image)
            
            random = np.random.randint(0, 100)
            if random < VALIDATION_PERCENTAGE:
                validation_image.append(image_value)
                validation_label.append(label_value)
            elif random < VALIDATION_PERCENTAGE + TEST_PERCENTAGE:
                test_image.append(image_value)
                test_label.append(label_value)
            else:
                train_image.append(image_value)
                train_label.append(label_value)
        
        label_value += 1
    
    state = np.random.get_state()
    train_image = np.random.shuffle(train_image)
    np.random.set_state(state)
    train_label = np.random.shuffle(train_label)
    
    return np.asarray([train_image, train_label, validation_image, validation_label, test_image, test_label])

def main():
    with tf.Session() as sess:
        precessed_image_data = create_image(sess)
        np.save(OUTPUT_FILE, precessed_image_data)
        
if __name__ == "__main__":
    main()