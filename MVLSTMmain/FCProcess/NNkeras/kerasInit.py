import tensorflow as tf
#from keras.backend import tensorflow_backend
#tensorflow v2以降はtensorfllowからkerasをインストール
from tensorflow.python.keras import backend as K

def kerasInit():
    #ConfigProtoはtensorflow v2 では非対応
    config = tf.compat.v1.ConfigProto(device_count={"GPU":0}, # GPUの数0に
                            gpu_options=tf.compat.v1.GPUOptions(allow_growth=True))
    session = tf.compat.v1.Session(config=config)
    #tensorflow_backend.set_session(session)
    K.set_session(session)
