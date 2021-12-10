import os

import requests
import numpy as np
import tensorflow as tf
from matplotlib.colors import hsv_to_rgb
from tqdm import tqdm

def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination)    

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)

class BigMorpheus:

    def __init__(self, params_path:str):

        self.model_path = params_path
        self.model = None


    @tf.autograph.experimental.do_not_convert
    def __call__(self, x):
        
        import tensorflow as tf
        if self.model is None:

            import tensorflow.keras.backend as K
            tf.autograph.set_verbosity(10, True)

            class QKEncoder(tf.keras.layers.Layer):
                """Query and Key embedding layer"""

                def __init__(self, filters:int, **kwargs):
                    super(QKEncoder, self).__init__(**kwargs)
                    self.filters = filters
                    self.conv = tf.keras.layers.Conv2D(filters, 1, padding="SAME")
                    self.bn = tf.keras.layers.BatchNormalization()
                    self.reshape = tf.keras.layers.Reshape([-1, filters])
                    self.l2_norm = tf.keras.layers.Lambda(lambda x: K.l2_normalize(x, axis=2))

                def call(self, inputs):
                    return self.l2_norm(
                        self.reshape(
                            self.bn(
                                self.conv(
                                    inputs
                                )
                            )
                        )
                    )

                # Adapted from:
                # https://github.com/tensorflow/tensorflow/blob/fcc4b966f1265f466e82617020af93670141b009/tensorflow/python/keras/layers/dense_attention.py#L483
                def get_config(self):
                    config = dict(
                        filters=self.filters,
                    )
                    base_config = super(QKEncoder, self).get_config()
                    return dict(list(base_config.items()) + list(config.items()))


            class VEncoder(tf.keras.layers.Layer):
                """Value embedding layer"""

                def __init__(self, filters:int, **kwargs):
                    super(VEncoder, self).__init__(**kwargs)
                    self.filters = filters
                    self.conv = tf.keras.layers.Conv2D(filters, 1, padding="SAME")
                    self.bn = tf.keras.layers.BatchNormalization()
                    self.reshape = tf.keras.layers.Reshape([-1, filters])
                    self.relu = tf.keras.layers.ReLU()

                def call(self, inputs):
                    bn = self.bn(self.conv(inputs))
                    encoding = self.relu(self.reshape(bn))

                    return encoding, bn

                # Adapted from:
                # https://github.com/tensorflow/tensorflow/blob/fcc4b966f1265f466e82617020af93670141b009/tensorflow/python/keras/layers/dense_attention.py#L483
                def get_config(self):
                    config = dict(
                        filters=self.filters,
                    )
                    base_config = super(VEncoder, self).get_config()
                    return dict(list(base_config.items()) + list(config.items()))


            class AdaptiveFastAttention(tf.keras.layers.Layer):
                """ Adaptive Fast Attention Layer.

                Based on:

                Real-time Semantic Segmentation with Fast Attention
                https://arxiv.org/pdf/2007.03815.pdf

                The change to the Fast Attention module is to vary the order of
                matrix multiplications operations according to the size of `n` and `c'`
                to minimize complexity.

                Args:
                    c_prime (int): The number of attention features in q and k
                """

                def __init__(self, c_prime:int=128, **kwargs):
                    super(AdaptiveFastAttention, self).__init__(**kwargs)
                    self.c_prime = c_prime

                @staticmethod
                def attention_qk_first(q, k, v):
                    qk = tf.keras.layers.Dot(axes=(2, 2))([q, k])
                    qkv = tf.keras.layers.Dot(axes=(2, 1))([qk, v])
                    return qkv

                @staticmethod
                def attention_kv_first(q, k, v):
                    kv = tf.keras.layers.Dot(axes=(1, 1))([k, v])
                    qkv = tf.keras.layers.Dot(axes=(2, 1))([q, kv])
                    return qkv


                def build(self, input_shape):
                    h = w = input_shape[1]
                    n = input_shape[1] * input_shape[2]
                    c = input_shape[-1]

                    n_coef = tf.constant(1 / n, dtype=tf.float32)

                    self.Q = QKEncoder(self.c_prime)
                    self.K = QKEncoder(self.c_prime)
                    self.V = VEncoder(c)

                    qkv_cost = (n**2 * self.c_prime) + (n**2 * c)
                    kvq_cost = (n * self.c_prime * c) * 2

                    self.qk_first = qkv_cost < kvq_cost

                    self.n_multiply = tf.keras.layers.Lambda(lambda x: n_coef * x)

                    if self.qk_first:
                        self.multiply_qkv = AdaptiveFastAttention.attention_qk_first
                    else:
                        self.multiply_qkv = AdaptiveFastAttention.attention_kv_first

                    self.square_qkv = tf.keras.layers.Reshape([h, w, c])
                    self.conv = tf.keras.layers.Conv2D(c, 3, padding="SAME")
                    self.bn = tf.keras.layers.BatchNormalization()
                    self.relu = tf.keras.layers.ReLU()

                    self.residual_add = tf.keras.layers.Add()

                def call(self, inputs):
                    q = self.Q(inputs)
                    k = self.K(inputs)
                    v, residual_v = self.V(inputs)

                    qkv = self.n_multiply(self.multiply_qkv(q, k, v))

                    out = self.relu(self.bn(self.conv(self.square_qkv(qkv))))

                    return self.residual_add([residual_v, out])

                # Adapted from:
                # https://github.com/tensorflow/tensorflow/blob/fcc4b966f1265f466e82617020af93670141b009/tensorflow/python/keras/layers/dense_attention.py#L483
                def get_config(self):
                    config = dict(c_prime=self.c_prime)
                    base_config = super(AdaptiveFastAttention, self).get_config()
                    return dict(list(base_config.items()) + list(config.items()))


            self.model = tf.keras.models.load_model(
                self.model_path,
                custom_objects={
                    "AdaptiveFastAttention" : AdaptiveFastAttention,
                    "QKEncoder" : QKEncoder,
                    "VEncoder" : VEncoder,
                    "tf" : tf
                },
            )

            
        model_out = tf.nn.softmax(
            self.model(
                tf.image.per_image_standardization(x[0])
            )
        ).numpy()

        return model_out




def get_morpheus():

    model_params = './3942b1630b424d83b28813c1f3337154-big-morpheus.h5' 
    if not os.path.exists(model_params):
        file_id = "1ZKGMQuSXFw-tMKMwSvqL7sg_0PUBkdXX"
        download_file_from_google_drive(file_id, model_params)
        
    return BigMorpheus(
        "./3942b1630b424d83b28813c1f3337154-big-morpheus.h5",
    )
  
    
def colorize_output(
        output, out_dir: str = None
    ) -> np.ndarray:
        """Makes a color image from the classification output.
        The colorization scheme is defined in HSV and is as follows:
        * Spheroid = Red
        * Disk = Blue
        * Irregular = Green
        * Point Source = Yellow
        The hue is set to be the color associated with the highest ranked class
        for a given pixel. The saturation is set to be the difference between the
        highest ranked class and the second highest ranked class for a given
        pixel. For example, if the top two classes have nearly equal values given
        by the classifier, then the saturation will be low and the pixel will
        appear more white. If the top two classes have very different
        values, then the saturation will be high and the pixel's color will be
        vibrant and not white. The value for a pixel is set to be 1-bkg, where
        bkg is value given to the background class. If the background class has
        a high value, then the pixel will appear more black. If the background
        value is low, then the pixel will take on the color given by the hue and
        saturation values.
        Args:
            data (dict): A dictionary containing the output from Morpheus.
            out_dir (str): a path to save the image in.
            hide_unclassified (bool): If true, black out the edges of the image
                                      that are unclassified. If false, show the
                                      borders as white.
        Returns:
            A [width, height, 3] array representing the RGB image.
        """
        red = 0.0  # spheroid
        blue = 0.7  # disk
        yellow = 0.18  # point source
        green = 0.3  # irregular

        shape = output.shape[:-1]

        colors = np.array([red, blue, green, yellow])
        morphs = output[..., :-1]
        ordered = np.argsort(-morphs, axis=-1)

        hues = np.zeros(shape)
        sats = np.zeros(shape)
        vals = 1 - output[..., -1]

        for i in tqdm(range(shape[0])):
            for j in range(shape[1]):
                hues[i, j] = colors[ordered[i, j, 0]]
                sats[i, j] = (
                    morphs[i, j, ordered[i, j, 0]] - morphs[i, j, ordered[i, j, 1]]
                )

        hsv = np.dstack([hues, sats, vals])
        rgb = hsv_to_rgb(hsv)

        if out_dir:
            png = (rgb * 255).astype(np.uint8)
            imageio.imwrite(os.path.join(out_dir, "colorized.png"), png)

        return rgb
