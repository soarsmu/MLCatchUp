import tensorflow as tf

def parse_image_test(image,label):
    image = tf.image.decode_jpeg(image,channels=3)
    label = tf.image.decode_jpeg(label,channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    label = tf.image.convert_image_dtype(label,tf.float32)
    shape = tf.shape(image)
    height_offset = tf.compat.v1.to_int32((shape[0] - 512) / 2)
    width_offset = tf.compat.v1.to_int32((shape[1] - 512) / 2)
    image = tf.image.crop_to_bounding_box(image, height_offset, width_offset, 512, 512)
    image = tf.image.crop_to_bounding_box(label, height_offset, width_offset, 512, 512)
    # image = tf.image.resize(image,[1200,900],method=tf.image.ResizeMethod.BICUBIC)
    # label = tf.image.resize(label,[1200,900],method=tf.image.ResizeMethod.BICUBIC)
    return image, label
