# crop to smaller images, divisible by 256x256
# this is important to make them the same. We can only crop because of the masks
# masks = colorful layer, training = the original image
import gc
import os

os.environ["SM_FRAMEWORK"] = "tf.keras"

import cv2
import numpy as np
import keras
import segmentation_models as sm
from keras.models import load_model
from keras.optimizers import Adam
from matplotlib import pyplot as plt
from patchify import patchify
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.callbacks import EarlyStopping
import tensorflow as tf

from simple_multi_unet_model import jacard_coef, f1_score, multi_unet_model, Attention_ResUNet, Attention_UNet, multi_unet_model_ELU


class SemanticModelTrainer:
    def __init__(self, data_path: str, patch_size: str = 256, focal_loss_strength: float = 0.5, cut_black_img: bool = True,
        use_regular_model: bool = True,
        filter_num: int = 32,
        batch_num:int = 16):
        self.scaler = MinMaxScaler()
        self.root_directory = data_path
        self.patch_size = patch_size

        self.lastBatch = 0
        self.focalLossStrength = focal_loss_strength

        self.cut_black_img = cut_black_img
        self.use_regular_model = use_regular_model
        self.filter_num = filter_num
        self.batch_num = batch_num

    # Read images from repsective 'images' subdirectory
    # As all images are of ddifferent size we have 2 options, either resize or crop
    # But, some images are too large and some small. Resizing will change the size of real objects.
    # Therefore, we will crop them to a nearest size divisible by 256 and then
    # divide all images into patches of 256x256x3.

    def create_datasets(self, batch: int):
        image_dataset = []

        for path, subdirs, files in os.walk(self.root_directory):
            # print(path)
            dirname = path.split(os.path.sep)[-1]
            if dirname == "images":  # Find all 'images' directories
                images = os.listdir(path)[
                    self.lastBatch : batch
                ]  # List of all image names in this subdirectory
                for i, image_name in enumerate(images):
                    if image_name.endswith(".tif"):  # Only read tif images...
                        image = cv2.imread(
                            path + "/" + image_name, 1
                        )  # Read each image as BGR

                        SIZE_X = (
                            image.shape[1] // self.patch_size
                        ) * self.patch_size  # Nearest size divisible by our patch size
                        SIZE_Y = (
                            image.shape[0] // self.patch_size
                        ) * self.patch_size  # Nearest size divisible by our patch size
                        image = Image.fromarray(image)
                        image = image.crop(
                            (0, 0, SIZE_X, SIZE_Y)
                        )  # Crop from top left corner
                        # image = image.resize((SIZE_X, SIZE_Y))  #Try not to resize for semantic segmentation
                        image = np.array(image)

                        # Large resized images
                        # Extract patches from each image
                        print("Now patchifying image:", path + "/" + image_name)
                        patches_img = patchify(
                            image,
                            (self.patch_size, self.patch_size, 3),
                            step=self.patch_size,
                        )  # Step=256 for 256 patches means no overlap

                        for i in range(patches_img.shape[0]):
                            for j in range(patches_img.shape[1]):
                                single_patch_img = patches_img[i, j, :, :]

                                # Use minmaxscaler instead of just dividing by 255.
                                single_patch_img = self.scaler.fit_transform(
                                    single_patch_img.reshape(
                                        -1, single_patch_img.shape[-1]
                                    )
                                ).reshape(single_patch_img.shape)

                                # single_patch_img = (single_patch_img.astype('float32')) / 255.
                                single_patch_img = single_patch_img[
                                    0
                                ]  # Drop the extra unecessary dimension that patchify adds.

                                image_dataset.append(single_patch_img)

        # Now do the same as above for masks
        # For this specific dataset we could have added masks to the above code as masks have extension png
        mask_dataset = []
        for path, subdirs, files in os.walk(self.root_directory):
            # print(path)
            dirname = path.split(os.path.sep)[-1]
            if dirname == "masks":  # Find all 'images' directories
                masks = os.listdir(path)[
                    self.lastBatch : batch
                ]  # List of all image names in this subdirectory
                for i, mask_name in enumerate(masks):
                    if mask_name.endswith(
                        ".png"
                    ):  # Only read png images... (masks in this dataset)
                        mask = cv2.imread(
                            path + "/" + mask_name, 1
                        )  # Read each image as Grey (or color but remember to map each color to an integer)
                        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
                        SIZE_X = (
                            mask.shape[1] // self.patch_size
                        ) * self.patch_size  # Nearest size divisible by our patch size
                        SIZE_Y = (
                            mask.shape[0] // self.patch_size
                        ) * self.patch_size  # Nearest size divisible by our patch size
                        mask = Image.fromarray(mask)
                        mask = mask.crop(
                            (0, 0, SIZE_X, SIZE_Y)
                        )  # Crop from top left corner
                        # mask = mask.resize((SIZE_X, SIZE_Y))  #Try not to resize for semantic segmentation
                        mask = np.array(mask)

                        # Extract patches from each image
                        print("Now patchifying mask:", path + "/" + mask_name)
                        patches_mask = patchify(
                            mask,
                            (self.patch_size, self.patch_size, 3),
                            step=self.patch_size,
                        )  # Step=256 for 256 patches means no overlap

                        for i in range(patches_mask.shape[0]):
                            for j in range(patches_mask.shape[1]):
                                self.single_patch_mask = patches_mask[i, j, :, :]
                                # single_patch_img = (single_patch_img.astype('float32')) / 255. #No need to scale masks, but you can do it if you want
                                self.single_patch_mask = self.single_patch_mask[
                                    0
                                ]  # Drop the extra unecessary dimension that patchify adds.
                                mask_dataset.append(self.single_patch_mask)



        image_dataset = np.array(image_dataset)
        mask_dataset = np.array(mask_dataset)
        self.lastBatch = batch

        # separate black and good images, then good images again and again until good and full black pictures are in equal measure
        if self.cut_black_img:
            # only try to cut if there is anything to cut
            if len(image_dataset) == 0 or len(mask_dataset) == 0:
                return image_dataset, mask_dataset

            # collect black (only negative) and images containing POI separately
            masks_to_keep = list()
            masks_of_black = list()
            for i, mask in enumerate(mask_dataset):
                is_full_black = np.all(mask == 0)
                if not is_full_black:
                    masks_to_keep.append(i)
                else:
                    masks_of_black.append(i)

            mask_positive_dataset = mask_dataset[masks_to_keep, :, :, :]
            image_positive_dataset = image_dataset[masks_to_keep, :, :, :]
            mask_negative_dataset = mask_dataset[masks_of_black, :, :, :]
            image_negative_dataset = image_dataset[masks_of_black, :, :, :]

            # augment each positive image by horizontal and vertical flip
            mask_augmented = list()
            image_augmented = list()
            for i in range(len(mask_positive_dataset)):
                # original
                mask_augmented.append(mask_positive_dataset[i])
                image_augmented.append(image_positive_dataset[i])
                # vertical flip
                mask_augmented.append(cv2.flip(mask_positive_dataset[i], 0))
                image_augmented.append(cv2.flip(image_positive_dataset[i], 0))
                # horizontal flip
                mask_augmented.append(cv2.flip(mask_positive_dataset[i], 1))
                image_augmented.append(cv2.flip(image_positive_dataset[i], 1))

            mask_dataset = list()
            image_dataset = list()
            num_positive = len(mask_augmented)

            for i in range(num_positive):
                mask_dataset.append(mask_augmented[i])
                image_dataset.append(image_augmented[i])

                mask_dataset.append(mask_negative_dataset[i])
                image_dataset.append(image_negative_dataset[i])

            mask_dataset = np.concatenate(mask_dataset)
            mask_dataset = np.reshape(mask_dataset, [-1, self.patch_size, self.patch_size, 3])
            image_dataset = np.concatenate(image_dataset)
            image_dataset = np.reshape(image_dataset, [-1, self.patch_size, self.patch_size, 3])


            '''FAILED EXPERIMENT!
            too much picture is culled from the training data. Model fails to recognize what the POI isn't. 
            Results increase the number of false positives
        
            '''

        '''
        # Sanity check, view few mages
        import random
        for i in range(25):
            image_number = random.randint(0, len(image_dataset))
            plt.figure(figsize=(12, 6))
            plt.subplot(121)
            plt.imshow(np.reshape(image_dataset[image_number], (self.patch_size, self.patch_size, 3)))
            plt.subplot(122)
            plt.imshow(np.reshape(mask_dataset[image_number], (self.patch_size, self.patch_size, 3)))
            plt.show()'''

        return image_dataset, mask_dataset





    ###########################################################################
    """
    RGB to HEX: (Hexadecimel --> base 16)
    This number divided by sixteen (integer division; ignoring any remainder) gives 
    the first hexadecimal digit (between 0 and F, where the letters A to F represent 
    the numbers 10 to 15). The remainder gives the second hexadecimal digit. 
    0-9 --> 0-9
    10-15 --> A-F

    Example: RGB --> R=201, G=, B=

    R = 201/16 = 12 with remainder of 9. So hex code for R is C9 (remember C=12)

    Calculating RGB from HEX: #3C1098
    3C = 3*16 + 12 = 60
    10 = 1*16 + 0 = 16
    98 = 9*16 + 8 = 152

    """
    # Convert HEX to RGB array
    # Try the following to understand how python handles hex values...
    #   a = int('3C', 16)  # 3C with base 16. Should return 60.
    #   print(a)
    # Do the same for all RGB channels in each hex code to convert to RGB

    def convert_RGB_to_label(self):
        # TODO turn it into a list of input and the labels are a list of output
        Building = "#3C1098".lstrip("#")
        self.Building = np.array(
            tuple(int(Building[i : i + 2], 16) for i in (0, 2, 4))
        )  # 60, 16, 152

        Land = "#8429F6".lstrip("#")
        self.Land = np.array(
            tuple(int(Land[i : i + 2], 16) for i in (0, 2, 4))
        )  # 132, 41, 246

        Road = "#6EC1E4".lstrip("#")
        self.Road = np.array(
            tuple(int(Road[i : i + 2], 16) for i in (0, 2, 4))
        )  # 110, 193, 228

        Vegetation = "FEDD3A".lstrip("#")
        self.Vegetation = np.array(
            tuple(int(Vegetation[i : i + 2], 16) for i in (0, 2, 4))
        )  # 254, 221, 58

        Water = "E2A929".lstrip("#")
        self.Water = np.array(
            tuple(int(Water[i : i + 2], 16) for i in (0, 2, 4))
        )  # 226, 169, 41

        Unlabeled = "#9B9B9B".lstrip("#")
        self.Unlabeled = np.array(
            tuple(int(Unlabeled[i : i + 2], 16) for i in (0, 2, 4))
        )  # 155, 155, 155

        self.label = self.single_patch_mask

    # Now replace RGB to integer values to be used as labels.
    # Find pixels with combination of RGB for the above defined arrays...
    # if matches then replace all values in that pixel with a specific integer
    def rgb_to_2D_label(self, label, input_labels):
        """
        Suply our labale masks as input in RGB format.
        Replace pixels with specific RGB values ...
        """
        label_seg = np.zeros(label.shape, dtype=np.uint8)

        # TODO turn it into a for loop

        for i in range(len(input_labels)):
            label_seg[np.all(label == input_labels[i].RGBColor, axis=-1)] = i
        # label_seg[np.all(label == input_labels, axis=-1)] = 0
        # label_seg[np.all(label == self.Land, axis=-1)] = 1
        # label_seg[np.all(label == self.Road, axis=-1)] = 2
        # label_seg[np.all(label == self.Vegetation, axis=-1)] = 3
        # label_seg[np.all(label == self.Water, axis=-1)] = 4
        # label_seg[np.all(label == self.Unlabeled, axis=-1)] = 5

        label_seg = label_seg[
            :, :, 0
        ]  # Just take the first channel, no need for all 3 channels

        return label_seg

    def label_masks(self, mask_dataset, input_layers):
        labels = []
        for i in range(mask_dataset.shape[0]):
            label = self.rgb_to_2D_label(mask_dataset[i], input_layers)
            labels.append(label)

        labels = np.array(labels)
        labels = np.expand_dims(labels, axis=3)

        print("Unique labels in label dataset are: ", np.unique(labels))
        return labels

        """
        # Another Sanity check, view few mages
        import random
        import numpy as np
        
        image_number = random.randint(0, len(image_dataset))
        plt.figure(figsize=(12, 6))
        plt.subplot(121)
        plt.imshow(image_dataset[image_number])
        plt.subplot(122)
        plt.imshow(labels[image_number][:, :, 0])
        plt.show()
        """
        ############################################################################



    def train(
        self,
        labels,
        image_dataset,
        model_name,
        categories,
        epochs: int = 15,
    ):
        n_classes = len(np.unique(categories))
        from keras.utils import to_categorical

        labels_cat = to_categorical(labels, num_classes=n_classes)  # one-hot encoding

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            image_dataset, labels_cat, test_size=0.20, random_state=42
        )

        #transform = A.Compose([
         #   A.HorizontalFlip(p=0.5),  # Horizontal flip with a 50% probability
         #   A.VerticalFlip(p=0.5),
         #   #A.RandomRotate90(p=0.5),  # Random 90-degree rotation
         #   A.Rotate(limit=(-90, 90), p=0.5),
         #   #A.RandomBrightnessContrast(p=0.2),  # Adjust brightness and contrast
         #   A.Normalize(),  # Normalize pixel values
        #])

        # augmented_x_train = []
        #augmented_y_train = []

        #for x, y in zip(self.X_train, self.y_train):
        #    augmented = transform(image=x, mask=y)
        #    augmented_x_train.append(augmented['image'])
        #    augmented_y_train.append(augmented['mask'])

        #augmented_x_train = np.array(augmented_x_train)
        #augmented_y_train = np.array(augmented_y_train)

        #######################################
        # Parameters for model
        # Segmentation models losses can be combined together by '+' and scaled by integer or float factor
        # set class weights for dice_loss
        # from sklearn.utils.class_weight import compute_class_weight


        #weights = compute_class_weight(class_weight='balanced', classes=np.unique(np.ravel(labels, order='C')),
                                       #y=np.ravel(labels, order='C'))

        # Extract class labels (last element of each picture entry)
        y_labels = np.ravel(labels, order='C').astype(int)

        # Calculate class frequencies
        pseudocount = 1
        class_counts = np.bincount(y_labels, minlength=n_classes) + pseudocount
        total_samples = len(y_labels)
        class_frequencies = class_counts / total_samples

        # Calculate class weights as the inverse of class frequencies
        weights = 1.0 / class_frequencies

        # Normalize class weights to sum to 1
        weights /= np.sum(weights)

        print(weights)

        '''
        weights = [
            0.1666,
            0.1666,
            0.1666,
            0.1666,
            0.1666,
            0.1666,
        ]  # We could adjust weights
        '''
        #tf.keras.utils.get_custom_objects().clear()

        #cross_entropy = sm.losses.CategoricalCELoss(class_weights=weights)
        #@tf.keras.utils.register_keras_serializable(package="my_package", name="custom_loss")
        def get_loss():
            dice_loss = sm.losses.DiceLoss(class_weights=weights)  #, class_indexes=[1])
            focal_loss = sm.losses.CategoricalFocalLoss(gamma=2, alpha=0.1666)  #, class_indexes=[1])
            total_loss = self.focalLossStrength * dice_loss + (self.focalLossStrength * focal_loss)
            return total_loss

        self.IMG_HEIGHT = self.X_train.shape[1]
        self.IMG_WIDTH = self.X_train.shape[2]
        self.IMG_CHANNELS = self.X_train.shape[3]

        metrics = ["accuracy", jacard_coef, f1_score]

        if os.path.exists("models/" + model_name + ".keras"):
            model = load_model("models/" + model_name + ".keras", compile=False)
        else:
            model = self.get_model(n_classes, regularModel=self.use_regular_model)

        opt = Adam(learning_rate=0.1)  # Please leave it like this
        model.compile(optimizer=opt, loss=get_loss(), metrics=metrics, loss_weights=weights, run_eagerly=True)

        # model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=metrics)
        #model.summary()

        early_stopping = EarlyStopping(
            monitor='loss',  # Monitor the validation loss
            patience=50,  # Number of epochs with no improvement after which training will be stopped
            restore_best_weights=True  # Restore the model weights to the best achieved during training
        )

        history1 = model.fit(
            self.X_train,
            self.y_train,
            batch_size=self.batch_num,
            verbose=1,
            epochs=epochs,
            shuffle=False,
            callbacks=[early_stopping]
        )
        del(self.X_train)
        del(self.y_train)
        #del(self.y_test)
        #del(self.X_test)


        '''
        #dataAugmentaion = ImageDataGenerator(rotation_range=0, zoom_range=0,fill_mode="nearest", shear_range=0, horizontal_flip=False,width_shift_range=0, height_shift_range=0)

        dataAugmentaion = ImageDataGenerator()
        history1 = model.fit_generator(
            dataAugmentaion.flow(self.X_train, self.y_train, batch_size = 16),
            validation_data = (self.X_test, self.y_test),
            steps_per_epoch = len(self.X_train) // 32,
            epochs = epochs)
        '''
        model.save("models/" + model_name + ".keras")
        del(model)
        print("Model", model_name, "is saved successfuly.")
        tf.keras.backend.clear_session()
        gc.collect()
        return history1

    def get_model(self, classes, regularModel: bool):
        if regularModel:
            return multi_unet_model_ELU(
                n_classes=classes,
                IMG_HEIGHT=self.IMG_HEIGHT,
                IMG_WIDTH=self.IMG_WIDTH,
                IMG_CHANNELS=self.IMG_CHANNELS,
                dropout_rate=0.2,
            )
        else:
            #return Attention_UNet(input_shape=(self.IMG_HEIGHT, self.IMG_WIDTH, self.IMG_CHANNELS), NUM_CLASSES=classes, start_filter_num=self.filter_num)
            return Attention_ResUNet(input_shape=(self.IMG_HEIGHT, self.IMG_WIDTH, self.IMG_CHANNELS), NUM_CLASSES=classes, start_filter_num=self.filter_num)



    # Minmaxscaler
    # With weights...[0.1666, 0.1666, 0.1666, 0.1666, 0.1666, 0.1666]   in Dice loss
    # With focal loss only, after 100 epochs val jacard is: 0.62  (Mean IoU: 0.6)
    # With dice loss only, after 100 epochs val jacard is: 0.74 (Reached 0.7 in 40 epochs)
    # With dice + 5 focal, after 100 epochs val jacard is: 0.711 (Mean IoU: 0.611)
    ##With dice + 1 focal, after 100 epochs val jacard is: 0.75 (Mean IoU: 0.62)
    # Using categorical crossentropy as loss: 0.71

    ##With calculated weights in Dice loss.
    # With dice loss only, after 100 epochs val jacard is: 0.672 (0.52 iou)

    ##Standardscaler
    # Using categorical crossentropy as loss: 0.677

    ############################################################
    # TRY ANOTHE MODEL - WITH PRETRINED WEIGHTS
    # Resnet backbone
    # BACKBONE = 'resnet34'
    # preprocess_input = sm.get_preprocessing(BACKBONE)

    # preprocess input
    # X_train_prepr = preprocess_input(X_train)
    # X_test_prepr = preprocess_input(X_test)

    # define model
    # model_resnet_backbone = sm.Unet(BACKBONE, encoder_weights='imagenet', classes=n_classes, activation='softmax')

    # compile keras model with defined optimozer, loss and metrics
    # model_resnet_backbone.compile(optimizer='adam', loss=focal_loss, metrics=metrics)
    # model_resnet_backbone.compile(optimizer='adam', loss='categorical_crossentropy', metrics=metrics)

    # print(model_resnet_backbone.summary())

    # history2 = model_resnet_backbone.fit(X_train_prepr,
    #                                    y_train,
    #                                   batch_size=16,
    #                                  epochs=5,
    #                                 verbose=1,
    #                                validation_data=(X_test_prepr, y_test))

    # Minmaxscaler
    # With weights...[0.1666, 0.1666, 0.1666, 0.1666, 0.1666, 0.1666]   in Dice loss
    # With focal loss only, after 100 epochs val jacard is:
    # With dice + 5 focal, after 100 epochs val jacard is: 0.73 (reached 0.71 in 40 epochs. So faster training but not better result. )
    ##With dice + 1 focal, after 100 epochs val jacard is:
    ##Using categorical crossentropy as loss: 0.755 (100 epochs)
    # With calc. weights supplied to model.fit:

    # Standard scaler
    # Using categorical crossentropy as loss: 0.74

    ###########################################################
    # plot the training and validation accuracy and loss at each epoch

    def show_metrics(self, history1):
        history = history1
        loss = history.history["loss"]
        val_loss = history.history["loss"]
        epochs = range(1, len(loss) + 1)
        # plt.plot(epochs, loss, "y", label="Training loss")
        # plt.plot(epochs, val_loss, "r", label="Validation loss")
        # plt.title("Training and validation loss")
        # plt.xlabel("Epochs")
        # plt.ylabel("Loss")
        # plt.legend()
        #plt.show()

        acc = history.history["jacard_coef"]
        val_acc = history.history["jacard_coef"]

        #plt.plot(epochs, acc, "y", label="Training IoU")
        # plt.plot(epochs, val_acc, "r", label="Validation IoU")
        # plt.title("Training and validation IoU")
        # plt.xlabel("Epochs")
        # plt.ylabel("IoU")
        # plt.legend()
        #plt.show()

    def predict(self, model_name):
        model = load_model("models/" + model_name + ".keras", compile=False)
        # custom_objects={'dice_loss_plus_2focal_loss': total_loss,
        #'jacard_coef': jacard_coef})

        # IOU
        y_pred = model.predict(self.X_test)
        y_pred_argmax = np.argmax(y_pred, axis=3)
        y_test_argmax = np.argmax(self.y_test, axis=3)

        # Using built in keras function for IoU
        from keras.metrics import MeanIoU

        n_classes = 2
        IOU_keras = MeanIoU(num_classes=n_classes)
        IOU_keras.update_state(y_test_argmax, y_pred_argmax)
        print("Mean IoU =", IOU_keras.result().numpy())

        # +predict on a few images
        import random

        test_img_number = random.randint(0, len(self.X_test)-1)
        test_img = self.X_test[test_img_number]
        ground_truth = y_test_argmax[test_img_number]
        # test_img_norm=test_img[:,:,0][:,:,None]
        test_img_input = np.expand_dims(test_img, 0)
        prediction = model.predict(test_img_input)
        predicted_img = np.argmax(prediction, axis=3)[0, :, :]

        cv2.imwrite("result.png", prediction)

        # plt.figure(figsize=(12, 8))
        # plt.subplot(231)
        # plt.title("Testing Image")
        # plt.imshow(test_img)
        # plt.subplot(232)
        # plt.title("Testing Label")
        # plt.imshow(ground_truth)
        # plt.subplot(233)
        # plt.title("Prediction on test image")
        # plt.imshow(predicted_img)
        # plt.show()
