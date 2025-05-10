import os
import tensorflow as tf
import random

IMAGE_SIZE = (224, 224)
IMAGES_PER_TFRECORD = 1000
OUTPUT_DIR = 'tfrecords'
INPUT_DIR = 'path/to/your/images'  # Change this

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def serialize_example(image_str, label):
    feature = {
        'image': _bytes_feature(image_str),
        'label': _int64_feature(label)
    }
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()

def get_image_paths_and_labels(data_dir):
    classes = sorted(os.listdir(data_dir))
    class_to_index = {cls: idx for idx, cls in enumerate(classes)}
    image_label_pairs = []
    for cls in classes:
        cls_folder = os.path.join(data_dir, cls)
        for fname in os.listdir(cls_folder):
            if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_label_pairs.append((os.path.join(cls_folder, fname), class_to_index[cls]))
    random.shuffle(image_label_pairs)
    return image_label_pairs, class_to_index

def write_tfrecord_shards(pairs, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for i in range(0, len(pairs), IMAGES_PER_TFRECORD):
        shard_path = os.path.join(output_dir, f'train-{i // IMAGES_PER_TFRECORD:05d}.tfrecord.gz')
        with tf.io.TFRecordWriter(shard_path, options='GZIP') as writer:
            for img_path, label in pairs[i:i + IMAGES_PER_TFRECORD]:
                image_bytes = tf.io.read_file(img_path)
                example = serialize_example(image_bytes.numpy(), label)
                writer.write(example)
        print(f'Wrote {shard_path}')

# Usage
pairs, class_map = get_image_paths_and_labels(INPUT_DIR)
write_tfrecord_shards(pairs, OUTPUT_DIR)
print(f"Done. Label map: {class_map}")