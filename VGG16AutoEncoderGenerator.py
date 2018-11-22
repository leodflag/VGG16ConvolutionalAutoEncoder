from keras.models import Sequential, Model
from keras.layers import Dense
from keras.layers.convolutional import Conv2D, MaxPooling2D, AveragePooling2D, UpSampling2D


class VGG16AutoEncoderGenerator:
    def __init__(self, input_shape, weight_file: str, number_of_layers_to_freeze_from_input_layer: int):
        self.encoder = VGG16EncoderGenerator(
            input_shape=input_shape, weight_file=weight_file,
            number_of_layers_to_freeze_from_input_layer=number_of_layers_to_freeze_from_input_layer
        ).encoder

        self.decoder = DecoderGenerator(self.encoder).decoder

        self.auto_encoder = Model(inputs=self.encoder.inputs, outputs=self.decoder)


class VGG16EncoderGenerator:
    def __init__(self, input_shape, weight_file: str,  number_of_layers_to_freeze_from_input_layer: int):
        self.encoder = self.__get_encoder(input_shape)  #  取得編碼器，輸入層數
        self.encoder.load_weights(weight_file)  #  取得權重檔案
        self.encoder = self.__freeze_weight(self.encoder, number_of_layers_to_freeze_from_input_layer) #  層數和固定的去固定的權重層
    #  取得卷積編譯層，輸入層數
    def __get_encoder(self, input_shape):
        #  Keras  Sequential 順序模型
        return Sequential([
            Conv2D(64, (3, 3), input_shape=input_shape, padding='same', activation='relu'),
            Conv2D(64, (3, 3), activation='relu', padding='same'),
            MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
            Conv2D(128, (3, 3), activation='relu', padding='same'),
            Conv2D(128, (3, 3), activation='relu', padding='same',),
            MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
            Conv2D(256, (3, 3), activation='relu', padding='same',),
            Conv2D(256, (3, 3), activation='relu', padding='same',),
            Conv2D(256, (3, 3), activation='relu', padding='same',),
            MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
            Conv2D(512, (3, 3), activation='relu', padding='same',),
            Conv2D(512, (3, 3), activation='relu', padding='same',),
            Conv2D(512, (3, 3), activation='relu', padding='same',),
            MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
            Conv2D(512, (3, 3), activation='relu', padding='same',),
            Conv2D(512, (3, 3), activation='relu', padding='same',),
            Conv2D(512, (3, 3), activation='relu', padding='same',),
            AveragePooling2D(pool_size=(2, 2), strides=(2, 2)),
        ])
    #  固定權重(編碼層，固定權重層)
    def __freeze_weight(self, encoder, number_of_layers_to_freeze_from_input_layer: int):
        for i in range(number_of_layers_to_freeze_from_input_layer):
            encoder.layers[i].trainable = False

        return encoder

#  解碼生成器
class DecoderGenerator:
    def __init__(self, encoder):
        self.decoder = self.___get_decoder(encoder)  #  藉由編碼層資訊取得解碼器

    def ___get_decoder(self, encoder):
        decoded_encoder = Dense(784, activation='relu')(encoder.output)

        decoded_encoder = Conv2D(
            filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'
        )(decoded_encoder)  #  2D輸入的上採樣層
        decoded_encoder = UpSampling2D(size=(2, 2))(decoded_encoder)

        decoded_encoder = Conv2D(
            filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'
        )(decoded_encoder)
        decoded_encoder = UpSampling2D(size=(2, 2))(decoded_encoder)

        decoded_encoder = Conv2D(
            filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'
        )(decoded_encoder)
        decoded_encoder = UpSampling2D(size=(2, 2))(decoded_encoder)

        decoded_encoder = Conv2D(
            filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'
        )(decoded_encoder)
        decoded_encoder = UpSampling2D(size=(2, 2))(decoded_encoder)

        decoded_encoder = Conv2D(
            filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(decoded_encoder)
        decoded_encoder = UpSampling2D(size=(2, 2))(decoded_encoder)

        decoder = Conv2D(
            filters=3, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='sigmoid'
        )(decoded_encoder)

        return decoder

