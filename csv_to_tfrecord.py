from absl import app, flags, logging
from pathlib import Path
import pandas as pd
import tensorflow as tf
import hashlib
import sys
from tqdm import tqdm
import io

from PIL import Image
from collections import namedtuple



FLAGS = flags.FLAGS
flags.DEFINE_string("src", "../data/train", "src folder")
flags.DEFINE_string("csv", None, "annotation csv file (def: 'src/annotations.csv'")
flags.DEFINE_string("classes", None, "class names file (def: 'src/classes.names'")
flags.DEFINE_string("dest", None, "output tfrecord file (def: 'src/src.tfrecord")

def int64_list_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def float_list_feature(value):
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def bytes_list_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

def split(df, group):
    data = namedtuple('data', ['filename', 'object'])
    gb = df.groupby(group)
    return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]

def create_tf_example(group, p_src, class_names):
    p_img_file = p_src / Path(group.filename)
    with tf.io.gfile.GFile(p_img_file.__str__(), 'rb') as fid:
        encoded_img = fid.read()
    key = hashlib.sha256(encoded_img).hexdigest()
    encoded_img_io = io.BytesIO(encoded_img)
    image = Image.open(encoded_img_io)
    width, height = image.size
    filename = group.filename
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []
    image_format = image.format.lower()

    for index, row in group.object.iterrows():
        # print(row)
        xmins.append(row['xmin'] / width)
        xmaxs.append(row['xmax'] / width)
        ymins.append(row['ymin'] / height)
        ymaxs.append(row['ymax'] / height)
        classes_text.append(class_names[row['class']].encode("utf8"))
        classes.append(row['class'])

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': int64_feature(height),
        'image/width': int64_feature(width),
        'image/filename': bytes_feature(filename.encode("utf8")),
        'image/source_id': bytes_feature(filename.encode("utf8")),
        'image/key/sha256': bytes_feature(key.encode('utf8')),
        'image/encoded': bytes_feature(encoded_img),
        'image/format': bytes_feature(image_format.encode('utf8')),
        'image/object/bbox/xmin': float_list_feature(xmins),
        'image/object/bbox/xmax': float_list_feature(xmaxs),
        'image/object/bbox/ymin': float_list_feature(ymins),
        'image/object/bbox/ymax': float_list_feature(ymaxs),
        'image/object/class/text': bytes_list_feature(classes_text),
        'image/object/class/label':int64_list_feature(classes),
    }))
    return tf_example

def get_classes(p_classes):
    with open(p_classes, "r") as file:
        class_names = [name.strip() for name in file.readlines()]
    return class_names

def csv_to_tfrecord(p_src, p_csv, p_out, p_classes):
    writer = tf.io.TFRecordWriter(p_out)
    examples = pd.read_csv(p_csv)
    grouped = split(examples, 'filename')
    class_names = get_classes(p_classes)

    logging.info("Building TFRecords from images:")
    for group in tqdm(grouped):
        tf_example = create_tf_example(group, p_src, class_names)
        writer.write(tf_example.SerializeToString())

    writer.close()

def get_paths_from_flags():
    p_src = Path(FLAGS.src)
    p_csv = FLAGS.csv
    if p_csv is None:
        p_csv = p_src / "annotations.csv"
    else:
        p_csv = Path(p_csv)

    p_classes = FLAGS.classes
    if p_classes is None:
        p_classes = p_src / "classes.names"
    else:
        p_classes = Path(p_classes)

    p_out = FLAGS.dest
    if p_out is None:
        p_out = p_src / "{}.tfrecord".format(p_src.stem)

    for p in [p_src, p_csv, p_classes]:
        if not p.exists():
            logging.error("{} does not exist".format(p))
            sys.exit()

    return {"src": str(p_src),
            "csv": str(p_csv),
            "out": str(p_out),
            "classes": str(p_classes)}

def main(argv):
    paths = get_paths_from_flags()
    log_str = "Data:" + "\n\t{:10}{}" * len(paths)
    logging.info(log_str.format(*[i for t in paths.items() for i in t]))
    csv_to_tfrecord(paths["src"], paths["csv"], paths["out"], paths["classes"])
    print('Successfully created TFRecords')

if __name__ == '__main__':
    app.run(main)