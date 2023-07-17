from keras.applications.vgg16 import decode_predictions, preprocess_input, VGG16
import numpy as np
import cv2
from PIL import Image

model = VGG16(weights='imagenet')

################################################

npy_image = np.load("acorn1/adversImg.npy").copy()                      # (1, 224, 224, 3) -1.0 to 1.0
npy_image = (npy_image / 2.0 + 0.5) * 255                                       # (1, 224, 224, 3) 0.0 to 255.0
npy_image = npy_image.reshape((1, 224, 224, 3))                                 # same
# Image.fromarray((npy_image[0]).astype(np.uint8)).save("npy_image.png")

yhat = model.predict(preprocess_input(npy_image))
pred_labels = decode_predictions(yhat, 1000)
print("LABEL OF THE NPY IMAGE IS: " + str(pred_labels[0][:3]))
print("----------------------------------")

################################################

loaded_image = cv2.imread("acorn1/adv_square.png")                                      # (224, 224, 3) 0 255
loaded_image = cv2.cvtColor(loaded_image, cv2.COLOR_BGR2RGB)                                    # (224, 224, 3) 0 255 rgb
loaded_image = (loaded_image.astype(np.float32) / 255.0) * 255.0                                # (224, 224, 3) 0.0 255.0
loaded_image = loaded_image.reshape((1, 224, 224, 3))                                           # (1, 224, 224, 3) 0.0 255.0
# Image.fromarray((loaded_image[0]).astype(np.uint8)).save("loaded_image.png")

yhat = model.predict(preprocess_input(loaded_image))
pred_labels = decode_predictions(yhat, 1000)
print("LABEL OF ADVERSARIAL 224*224 IMAGE IS: " + str(pred_labels[0][:3]))
print("----------------------------------")

################################################

final_image = cv2.imread("acorn1/adv_original.png")   
final_image = cv2.cvtColor(final_image, cv2.COLOR_BGR2RGB)                                  # (374, 500, 3) 0 255 rgb
final_image = cv2.resize(final_image, (224, 224), interpolation=cv2.INTER_LANCZOS4)         # (224, 224, 3) 0 255
final_image = (final_image.astype(np.float32) / 255.0) * 255.0                              # (224, 224, 3) 0.0 255.0
final_image = final_image.reshape((1, 224, 224, 3))                                         # (1, 224, 224, 3) 0.0 255.0
# Image.fromarray((final_image[0]).astype(np.uint8)).save("final_image.png")

yhat = model.predict(preprocess_input(final_image))
pred_labels = decode_predictions(yhat, 1000)
print("LABEL OF ORGINAL RECTANGULAR IMAGE IS: " + str(pred_labels[0][:3]))
print("----------------------------------")

################################################
