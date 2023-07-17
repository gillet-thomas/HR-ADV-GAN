from __future__ import print_function
import tensorflow

from tensorflow.keras import backend as K
from tensorflow.keras import layers, Model
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.metrics import binary_accuracy
from tensorflow.keras.layers import Input, Dense, Dropout, Flatten, Activation, LeakyReLU, Conv2D, Conv2DTranspose, BatchNormalization
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.applications.vgg16 import decode_predictions

import os, cv2
from PIL import Image
import numpy as np
from numpy.random import seed

class DCGAN():

    def __init__(self):
        # input image dimensions
        inputs = Input(shape=(224, 224, 3))
        
        optimizer_g = Adam(0.0002)
        optimizer_d = SGD(0.01)

        # Build generator
        generator = self.build_generator2(inputs)
        self.G = Model(inputs, generator)
        self.G._name = 'Generator'
        # self.G.summary()

        # Build discriminator and train it
        discriminator = self.build_discriminator2(self.G(inputs))
        self.D = Model(inputs, discriminator)
        self.D.compile(loss=tensorflow.keras.losses.binary_crossentropy, optimizer=optimizer_d, metrics=[self.custom_acc])
        self.D._name = 'Discriminator'
        # self.D.summary()
        
        # We use VGG16 trained with ImageNet dataset.
        self.target = VGG16(weights='imagenet')
        self.target.trainable = False

        # Build GAN: stack generator, discriminator and target
        img = (self.G(inputs) / 2 + 0.5) * 255  # image's pixels will be between [0, 255]

        # Image is now preprocessed before being fed to VGG16
        self.stacked = Model(inputs=inputs, outputs=[self.G(inputs), 
                                self.D(inputs), self.target(preprocess_input(img))])

        self.stacked.compile(loss=[self.generator_loss, tensorflow.keras.losses.binary_crossentropy,
                                   tensorflow.keras.losses.categorical_crossentropy], optimizer=optimizer_g)
        self.stacked.summary()

    def generator_loss(self, y_true, y_pred):
        return K.mean(K.maximum(K.sqrt(K.sum(K.square(y_pred - y_true), axis=-1)) - 0.3, 0), axis=-1)  # Hinge loss

    def custom_acc(self, y_true, y_pred):
        return binary_accuracy(K.round(y_pred), K.round(y_true))

    # Basic classification model
    def build_discriminator(self, inputs):
        D = Conv2D(32, 4, strides=(2, 2))(inputs) # Downsample the image by 2
        D = LeakyReLU()(D)  # Activation function
        D = Dropout(0.4)(D) # Dropout layer for weight regularization

        D = Conv2D(64, 4, strides=(2, 2))(D) # 64 filters, 4x4 kernel size, stride 2
        D = BatchNormalization()(D)
        D = LeakyReLU()(D)
        D = Dropout(0.4)(D)
        D = Flatten()(D)

        D = Dense(64)(D)
        D = BatchNormalization()(D)
        D = LeakyReLU()(D)
        D = Dense(1, activation='sigmoid')(D)
        return D
    
    def build_discriminator2(self, inputs):
        D = Conv2D(64, 3, strides=(2, 2), padding='same')(inputs) # Downsample the image by 2
        D = LeakyReLU()(D)  # Activation function
        # D = Dropout(0.5)(D) # Dropout layer for weight regularization

        D = Conv2D(128, 3, strides=(2, 2), padding='same')(D) # Downsample the image by 2
        D = BatchNormalization()(D)
        D = LeakyReLU()(D)  # Activation function
        # D = Dropout(0.5)(D) # Dropout layer for weight regularization

        D = Conv2D(256, 3, strides=(2, 2), padding='same')(D) # Downsample the image by 2
        D = BatchNormalization()(D)
        D = LeakyReLU()(D)  # Activation function
        # D = Dropout(0.5)(D) # Dropout layer for weight regularization

        D = Conv2D(512, 3, strides=(2, 2), padding='same')(D) # Downsample the image by 2
        D = BatchNormalization()(D)
        D = LeakyReLU()(D)  # Activation function
        # D = Dropout(0.5)(D) # Dropout layer for weight regularization

        D = Conv2D(512, 3, strides=(2, 2), padding='same')(D) # Downsample the image by 2
        D = BatchNormalization()(D)
        D = LeakyReLU()(D)  # Activation function
        # D = Dropout(0.5)(D) # Dropout layer for weight regularization

        D = Flatten()(D)
        D = Dense(512)(D)               # Withtout that layer the acorn decreases but rhino plateaus
        D = LeakyReLU()(D)
        # D = Dropout(0.5)(D) # Dropout layer for weight regularization
        D = Dense(1, activation='sigmoid')(D)

        return D

    def build_generator(self, generator_inputs):

        print("Generator input is: ", generator_inputs.shape)  # (none, 224, 224, 3)

        # c3s1-8
        G = Conv2D(8, 3, padding='same')(generator_inputs)
        G = BatchNormalization()(G)
        G = Activation('relu')(G)

        print("Layer 1 is: ", G.shape)  # (none, 374, 500, 64)

        # d16
        G = Conv2D(16, 3, strides=(2, 2), padding='same')(G)
        G = BatchNormalization()(G)
        G = Activation('relu')(G)

        print("Layer 2 is: ", G.shape)  # (none, 187, 250, 128)

        # d32
        G = Conv2D(32, 3, strides=(2, 2), padding='same')(G)
        G = BatchNormalization()(G)
        G = Activation('relu')(G)
        residual = G

        print("Layer 3 is: ", G.shape)  # (none, 94, 125, 256)
        
        # four r32 blocks
        for _ in range(4):
            G = Conv2D(32, 3, padding='same')(G)
            G = BatchNormalization()(G)
            G = Activation('relu')(G)
            G = Conv2D(32, 3, padding='same')(G)
            G = BatchNormalization()(G)
            G = layers.add([G, residual])
            residual = G

        print("Layer 4 is: ", G.shape)  # (none, 94, 125, 256)

        # u16
        G = Conv2DTranspose(16, 3, strides=(2, 2), padding='same')(G)
        G = BatchNormalization()(G)
        G = Activation('relu')(G)

        print("Layer 5 is: ", G.shape)  # (none, 187, 250, 128)

        # u8
        G = Conv2DTranspose(8, 3, strides=(2, 2), padding='same')(G)
        G = BatchNormalization()(G)
        G = Activation('relu')(G)

        print("Layer 6 is: ", G.shape)  # (none, 374, 500, 64)


        G = Conv2D(3, 3, padding='same')(G)
        # G = BatchNormalization()(G)
        G = Activation('relu')(G)

        print("Layer 7 is: ", G.shape)  # (none, 374, 500, 3)

        ## Add the generated noise to the original image
        ## 2.55 = epsilon, each pixel can be tweaked  by + or -2.5
        ## The more we increase the epsilon the faster the adv image is generated but the more noise can be seen as well
        G = layers.add([G*2.55/255, generator_inputs])  
        # Multiplying 2.55/255 = 0.01 will drastically reduce the magnitude of the noise, making it invisible. You can increase                                                 # this value, which would help advGAN_HR to generate the adversarial image more easily, but the visual quality will decrease. 
        return G

    def build_generator2(self, generator_inputs):

        print("Generator input shape is: ", generator_inputs.shape) # (none, 448, 448, 3)

        # c3s1-8
        G = Conv2D(64, 3, padding='same')(generator_inputs)
        G = BatchNormalization()(G)
        G = Activation('relu')(G)

        print("Layer 1 is: ", G.shape)  # (none, 448, 448, 64)

        # d16
        G = Conv2D(128, 3, strides=(2, 2), padding='same')(G)
        G = BatchNormalization()(G)
        G = Activation('relu')(G)

        print("Layer 2 is: ", G.shape) # (none, 224, 224, 128)

        # d32
        G = Conv2D(256, 3, strides=(2, 2), padding='same')(G)
        G = BatchNormalization()(G)
        G = Activation('relu')(G)
        residual = G

        print("Layer 3 is: ", G.shape) # (none, 112, 112, 256)
        
        # four r32 blocks
        for _ in range(4):
            G = Conv2D(256, 3, padding='same')(G)
            G = BatchNormalization()(G)
            G = Activation('relu')(G)
            G = Conv2D(256, 3, padding='same')(G)
            G = BatchNormalization()(G)
            G = layers.add([G, residual])
            residual = G

        print("Layer 4 is: ", G.shape) # (none, 112, 112, 256)

        # u16
        G = Conv2DTranspose(128, 3, strides=(2, 2), padding='same')(G)
        G = BatchNormalization()(G)
        G = Activation('relu')(G)

        print("Layer 5 is: ", G.shape) # (none, 224, 224, 128)

        # u8
        G = Conv2DTranspose(64, 3, strides=(2, 2), padding='same')(G)
        G = BatchNormalization()(G)
        G = Activation('relu')(G)

        print("Layer 6 is: ", G.shape) # (none, 448, 448, 64)

        G = Conv2D(3, 3, padding='same')(G)
        G = BatchNormalization()(G)
        G = Activation('tanh')(G)

        print("Layer 7 is: ", G.shape) # (none, 448, 448, 3)

        ## Add the generated noise to the original image
        ## 2.25 = epsilon, each pixel can be tweaked  by + or -2.5
        ## The more we increase the epsilon the faster the adv image is generated but the more noise can be seen as well
        G = layers.add([G*2.25/255, generator_inputs])  
        # Multiplying 2.25/255 = 0.01 will drastically reduce the magnitude of the noise, making it invisible. You can increase 
        # this value, which would help advGAN_HR to generate the adversarial image more easily, but the visual quality will decrease. 

        print("Layer 8 is: ", G.shape) # (none, 448, 448, 3)

        return G
    
    # Original method from the paper
    def train_discriminator(self, x_batch, Gx_batch):
        # for each batch:
        # predict noise on generator: G(z) = batch of fake images
        # train fake images on discriminator: D(G(z)) = update D params per D's classification for fake images
        # train real images on disciminator: D(x) = update D params per classification for real images
        self.D.trainable = True
        d_loss_real = self.D.train_on_batch(x_batch, np.random.uniform(0.9, 1.0, size=(len(x_batch), 1)))  # real=1, positive label smoothing
        d_loss_fake = self.D.train_on_batch(Gx_batch, np.zeros((len(Gx_batch), 1)))  # fake=0
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
        return d_loss  # (loss, accuracy) tuple

    def train_generator(self, x_batch):
        # for each batch:
        # train fake images on discriminator: D(G(z)) = update G params per D's classification for fake images
        arr = np.zeros(1000)
        arr[targx] = 1  # targx is index of the target class (rhinoceros beetle)
        full_target = np.tile(arr, (len(x_batch), 1))

        # Update only G params
        self.D.trainable = False
        self.target.trainable = False

        # input x_batch is the real images that will be used to generate the adversarial images
        # target 1 = target value for generator. It should output the real image for all generated images
        # target 2 = target value for discriminator. It should output 1 for all generated images
        # target 3 = target value for VGG16. It should output the target class for all generated images
        stacked_loss = self.stacked.train_on_batch(x_batch, [x_batch, np.ones((len(x_batch), 1)), full_target])
        return stacked_loss  # (generator loss, hinge loss, gan loss, adv loss)

    def train_GAN(self):

        # Create a directory to save the adversarial images
        if not os.path.exists(directory):
            os.makedirs(directory)

        # Load and preprocess image
        img = cv2.imread(path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # (448, 448, 3) between 0 and 255
        x_train = cv2.resize(img_rgb, (224, 224), interpolation=cv2.INTER_LANCZOS4)  # (448, 448, 3) - 0 to 255
        x_train = np.expand_dims(x_train, axis=0) # (1, 448, 448, 3) - 0 to 255
        x_train = np.array(x_train, dtype=np.float32) # (1, 448, 448, 3) - 0.0 to 255.0
        x_train = (x_train * 2. / 255 - 1).reshape(len(x_train), 224, 224, 3)  # (1, 448, 448, 3) - -1.0 to 1.0

        epochs = EPOCH
        for epoch in range(epochs):
            print("===========================================")
            print("EPOCH: ", epoch)
            Gx = self.G.predict(x_train)         # Gx (1, 224, 224, 3) from -1.0086238 to 1.0082918 ; x_train (1, 224, 224, 3)
            Gx = np.clip(Gx, -1, 1)              # Gx (1, 224, 224, 3) from -1.0 to 1.0
            
            (d_loss, d_acc) = self.train_discriminator(x_train, Gx)
            (g_loss, hinge_loss, gan_loss, adv_loss) = self.train_generator(x_train)

            print("Discriminator -- Loss:%f\tAccuracy:%.2f%%\nGenerator -- Loss:%f\nTarget Loss: %f\nGAN Loss: %f" % (d_loss, d_acc * 100., g_loss, adv_loss, gan_loss))
            np.save(directory + '/adversImg.npy', Gx) # (1, 224, 224, 3) -1.0 to 1.0

            # Save the 224*224 adversarial image
            adv_npy = np.load(directory + "/adversImg.npy").copy()
            adv_square = (adv_npy / 2.0 + 0.5) * 255            # (224, 224, 3) 0.0 to 255.0
            adv_square = adv_square.reshape((1, 224, 224, 3))   # (1, 224, 224, 3) 0.0 to 255.0
            Image.fromarray((adv_square[0]).astype(np.uint8)).save(directory + "/adv_square.png", 'png')

            # Extract noise and and scale it up to the original image size
            img_original = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)  # (374, 500, 3) 0 to 255
            img_original_224 = cv2.resize(img_original, (224, 224), interpolation=cv2.INTER_LANCZOS4) # (224, 224, 3) 0 to 255
            height, width = img_original.shape[:2]      # (374, 500)
            noise = adv_square[0] - img_original_224    # (224, 224, 3) from -255.0 to 255.0
            noise_scaled = cv2.resize(noise, (width, height), interpolation=cv2.INTER_LANCZOS4)  # (374, 500, 3) from -255.0 to 255.0
            
            # Create adv_original image = original image + scaled noise
            adv_original = img_original + noise_scaled      # (374, 500, 3) from -255.0 to 500.0 ?
            adv_original = np.clip(adv_original, 0, 255)    # (374, 500, 3) 0.0 255.0
            Image.fromarray(adv_original.astype(np.uint8)).save(directory + "/adv_original.png")

            # Re-open original image, scale it down to 224x224 for VGG and save it
            adv_original_loaded = cv2.imread(directory + "/adv_original.png") 
            adv_original_loaded = cv2.cvtColor(adv_original_loaded, cv2.COLOR_BGR2RGB)                               # (374, 500, 3) 0 255 rgb
            adv_original_loaded = cv2.resize(adv_original_loaded, (224, 224), interpolation=cv2.INTER_LANCZOS4)      # (224, 224, 3) 0 255
            adv_original_loaded = (adv_original_loaded.astype(np.float32) / 255.0) * 255.0                           # (224, 224, 3) 0.0 255.0
            adv_original_loaded = adv_original_loaded.reshape((1, 224, 224, 3))    
            Image.fromarray((adv_original_loaded[0]).astype(np.uint8)).save(directory + "/adv_original_224.png")
            
            # Predict the class of the adv_original image
            yhat = self.target.predict(preprocess_input(adv_original_loaded))
            pred_labels = decode_predictions(yhat, 1000)
            
            # Print max probability class and target class probability
            pred_max = pred_labels[0][0]
            for label in pred_labels[0]: 
                if label[1] == target:
                    print(label[1], label[2])
            print(pred_max[1], pred_max[2])

            # If target class is predicted or last epoch, break
            if (np.argmax(yhat, axis=1) == targx and pred_max[2] >= TARGET_ACCURACY) or epoch == epochs-1:
                # Img = Image.fromarray((adv_original[0]).astype(np.uint8))
                # filename = f"final_{epoch}.png"
                # Img.save(directory + "/" + filename, 'png')  
                print("ADVERSARIAL IMAGE FOUND! AFTER %d EPOCHS" % epoch)           
                break


EPOCH = 10000
TARGET_ACCURACY = 0.60

directory = "acorn1"
path = "data/acorn1.JPEG"
target = 'rhinoceros_beetle'
targx = 306


if __name__ == '__main__':
    seed(5) 
    tensorflow.random.set_seed(1)
    dcgan = DCGAN()
    dcgan.train_GAN()
