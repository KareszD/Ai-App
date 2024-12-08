# https://youtu.be/HrGn4uFrMOM
"""
Prediction using smooth tiling as descibed here...

https://github.com/Vooban/Smoothly-Blend-Image-Patches


"""
import os
import shutil
import gc

import cv2

# import keras.engine.functional as functional
import numpy as np
import rasterio
import tensorflow as tf

# import segmentation_models as sm
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

import DotResultInference
import id_to_SHP
import identificationModelTrainer
import identificationModelTrainer as imt
import Tiff_to_JPG
from PNG_to_SHP import collect_shps_to_one, save_as_single_shapefile
from SemanticModelTrainer import SemanticModelTrainer
from DotResultInference import CreateInference
from smooth_tiled_predictions import predict_img_with_smooth_windowing
from ultralytics.data.utils import autosplit

# os.environ["SM_FRAMEWORK"] = "tf.keras"


class Predictor:
    def __init__(
        self, Path: str, OutputDir: str, classNum: int = 6, patchSize: int = 256, IsOutputPOI: bool = False, secondaryPatchSize: int = 128
    ):
        self.classes = classNum
        self.patchSize = patchSize
        self.secondaryPatchSize = secondaryPatchSize
        self.DataPath = Path
        self.outputPath = OutputDir
        self.IsPOI = IsOutputPOI

        physical_devices = tf.config.list_physical_devices("GPU")
        print(physical_devices)
        if len(physical_devices) > 0:
            tf.config.experimental.set_memory_growth(physical_devices[0], True)

            config = tf.compat.v1.ConfigProto()
            config.gpu_options.allow_growth = True
            self.sess = tf.compat.v1.Session(config=config)

    def Start(self, modelPath: str, input_labels: list, needsSpliting: bool = True, is_POI: bool = True):
        if needsSpliting:
            con = Tiff_to_JPG.Converter(
                self.DataPath, self.outputPath, (1500, 1000), input_labels
            )
            arr = con.convert()  # Holds the middle-sized images

        ShpCollectPaths = []
        id_index = 0
        for path in arr:  # TODO Could be turned int paralel execution
            tf.keras.backend.clear_session()

            new_id_index = self.predictMask(
                input_labels, modelPath, path, id_counter=id_index
            )

            if os.path.isfile(f"{os.path.abspath(os.getcwd())}/Results/{path[:-4]}/SHP.shp"):
                ShpCollectPaths.append(
                    f"{os.path.abspath(os.getcwd())}/Results/{path[:-4]}"
                )
                id_index = new_id_index
            else:
                print(f"{os.path.abspath(os.getcwd())}/Results/{path[:-4]} is not a valid file")

        if len(ShpCollectPaths) > 0:
            return collect_shps_to_one(ShpCollectPaths, self.outputPath, is_POI=is_POI)

        return None
    """
    def TrainSemanticModel(
        self,
        train_data_path: str,
        intermediate_res_path,
        labels: list,
        batch: int,
        epochs: int,
        model_name: str,
        focal_loss_strength: float = 0.5,
        gausKernel: int = 7,
        gausSigma: float = 0,
        clipLimit: float = 0.5,
        tileGridSize: int = 20,
        create_temp_files: bool = True,
        marker_size: int = 16,
        cut_black_img: bool = True,
        use_regular_model: bool = True,
        filter_num: int = 32,
        batch_num:int = 16
    ):
        gc.enable()
        if create_temp_files:
            converter = Tiff_to_JPG.Converter(
                train_data_path, intermediate_res_path, (250, 250), labels, gausKernel, gausSigma, clipLimit, tileGridSize, marker_size=marker_size
            )
            _ = converter.convert(True)

        smt = SemanticModelTrainer(intermediate_res_path,
                                    self.patchSize,
                                    focal_loss_strength=focal_loss_strength,
                                    cut_black_img=cut_black_img,
                                    use_regular_model=use_regular_model,
                                    filter_num=filter_num,
                                    batch_num=batch_num
        )

        while True:
            tf.keras.backend.clear_session()
            gc.collect()
            #self.sess.run()
            images = None
            masks = None
            labeled_images = None

            images, masks = smt.create_datasets(batch)
            if len(images) == 0 or len(masks) == 0:
                break

            batch += batch
            # smt.convert_RGB_to_label()

            labeled_images = smt.label_masks(masks, labels)

            history = smt.train(
                labeled_images,
                images,
                model_name,
                labels,
                epochs,
            )

            smt.show_metrics(history)
            smt.predict(model_name)

        placeholder = self.DataPath
        self.DataPath = "G:\SULI\Projectmunka\git\\NeuralNetworkForFoliageDetection\Data\kutyaBigTest" #"G:\\SULI\\Projectmunka\\git\\NeuralNetworkForFoliageDetection\\Data\\SmolDog"
        res = self.Start(f"models/{model_name}.keras", labels, is_POI=True)

        is_inference = True
        if is_inference and res is not None:
            _, _, _, result_txt = CreateInference("G:\SULI\Projectmunka\git\\NeuralNetworkForFoliageDetection\Data\kutyaBigTest",
                                                  'Data/kutya/shapes', res)
            file = open("results.txt", "a")
            file.write(f"model named: {model_name} results: \r\n{result_txt}\r\n---------------------------------\r\n")
            file.close()

        self.DataPath = placeholder
    """

        
    def TrainIdentificationModel(self,
            train_data_path: str,
            intermediate_res_path,
            labels: list,
            batch: int,
            epochs: int,
            model_name: str,
            create_temp_files=True,
            gausKernel: int = 7,
            gausSigma: float = 0,
            clipLimit: float = 0.5,
            tileGridSize: int = 20,
            isPopulate=False,
            isUseBackGround=False,
            label_balance=0.1
                                ):
        if create_temp_files:
            #shutil.rmtree('tmp')
            converter = Tiff_to_JPG.Converter(
                train_data_path, 'val_tmp', (self.patchSize, self.patchSize), labels, gausKernel, gausSigma, clipLimit, tileGridSize, label_balance=label_balance, use_preprocess=True
            )
            _ = converter.convert(isTraining=True, isPopulate=False, isUseBackGround=False)

            # preserve shapefiles so we can continue with train.
            os.makedirs('val_tmp/shapes')
            for path, subdirs, files in os.walk(f'{train_data_path}/shapes'):
                for file in files:
                    shutil.copyfile(f"{path}/{file}", f'val_tmp/shapes/{file}')

            # convert to 64x64
            converter = Tiff_to_JPG.Converter(
                'val_tmp', intermediate_res_path, (self.secondaryPatchSize, self.secondaryPatchSize), labels, label_balance=label_balance, use_preprocess=False)
            _ = converter.convert(isTraining=True, isPopulate=isPopulate, isUseBackGround=isUseBackGround)

            # Remove middle images
            shutil.rmtree('val_tmp')

        idModel = identificationModelTrainer.YOLOTrainer(self.secondaryPatchSize, batch=batch, epoch=epochs, name=model_name)
        idModel.train_model()#idModel.tune_model()

        return idModel

    def validate_identification(self, modelPath: str, batch: int, data_path: str, input_labels: list, temp_folder:str = "test_tmp", model=None, needs_splitting=True):
        if needs_splitting:
            # convert to middle sized images
            converter = Tiff_to_JPG.Converter(
                data_path, 'val_tmp', (self.patchSize, self.patchSize), input_labels, use_preprocess=True)
            _ = converter.convert(isTraining=True, isPopulate=False, isUseBackGround=True)

            #preserve shapefiles so we can continue with train.
            os.makedirs('val_tmp/shapes')
            for path, subdirs, files in os.walk(f'{data_path}/shapes'):
                for file in files:
                    shutil.copyfile(f"{path}/{file}", f'val_tmp/shapes/{file}')

            # convert to 64x64
            converter = Tiff_to_JPG.Converter(
                'val_tmp', temp_folder, (self.secondaryPatchSize, self.secondaryPatchSize), input_labels, use_preprocess=False)
            _ = converter.convert(isTraining=True, isPopulate=False, isUseBackGround=True)

            #Remove middle images
            shutil.rmtree('val_tmp')

            autosplit(f"{temp_folder}/images", weights=(0.0, 0.0, 1.0), annotated_only=False)

        if model is None:
            model = identificationModelTrainer.YOLOTrainer(self.secondaryPatchSize, batch=batch, name=modelPath)
            model.validate(modelPath)

    def predict_identification(self, modelPath: str, data_path: str, input_labels:list, model = None, needs_splitting=True):
        image_paths = []
        if needs_splitting:
            con = Tiff_to_JPG.Converter(
                    self.DataPath, self.outputPath, (self.patchSize, self.patchSize), input_labels, use_preprocess=True
            )
            image_paths = con.convert(isPopulate=False, isTraining=False, isUseBackGround=False)  # Holds the middle-sized images
        else:
            for root, dirs, files in os.walk(data_path):
                for file in files:
                    if file.endswith(".tiff") or file.endswith(".tif"):
                        image_paths.append(os.path.join(root, file))

        images_boxes = None
        if model is None:
            idModel = identificationModelTrainer.YOLOTrainer(self.patchSize, name=modelPath)
            images_boxes = idModel.predict(model_name=modelPath, outputPath=self.outputPath, images=image_paths, smallImgsz=self.secondaryPatchSize)

        else:
            images_boxes = model.predict(outputPath=self.outputPath, images=image_paths, smallImgsz=self.secondaryPatchSize)

        if images_boxes is not None:
            proc = id_to_SHP.bboxProcessor(images_boxes)
            result = proc.process_results()

            shapes = "Data\\" + self.DataPath.split('\\')[-2] + "\\shapes"

            _, _, _, result_txt = DotResultInference.CreateInference(data_path, shapes, result)
            file = open("results.txt", "a")
            file.write(
                f"{modelPath} results on data {self.DataPath}: \r\n{result_txt}\r\n---------------------------------\r\n")
            file.close()
        else:
            print("NO PREDICTION WAS MADE TO ANY OF THE IMAGES")


    def predictMask(
        self,
        input_labels: list,
        modelPath: str = "models/satellite_standard_unet_5epochs.hdf5",
        dataP: str = "split00.png",
        id_counter=0,
    ):
        scaler = MinMaxScaler()

        img = cv2.imread(dataP, 1)

        model = load_model(modelPath, compile=False)  # , safe_mode=False)
        # size of patches
        patch_size = self.patchSize

        # Number of classes
        n_classes = self.classes

        #################################################################################
        # Predict patch by patch with no smooth blending
        ###########################################
        """
        SIZE_X = (img.shape[1] // patch_size) * patch_size  # Nearest size divisible by our patch size
        SIZE_Y = (img.shape[0] // patch_size) * patch_size  # Nearest size divisible by our patch size
        large_img = Image.fromarray(img)
        large_img = large_img.crop((0, 0, SIZE_X, SIZE_Y))  # Crop from top left corner
        # image = image.resize((SIZE_X, SIZE_Y))  #Try not to resize for semantic segmentation
        large_img = np.array(large_img)

        if ((np.array(large_img.shape) - (patch_size, patch_size, 3)) < 0).any():
            return

        patches_img = patchify(large_img, (patch_size, patch_size, 3), step=patch_size)  # Step=256 for 256 patches means no overlap
        patches_img = patches_img[:, :, 0, :, :, :]

        patched_prediction = []
        for i in range(patches_img.shape[0]):
            for j in range(patches_img.shape[1]):
                single_patch_img = patches_img[i, j, :, :, :]

                # Use minmaxscaler instead of just dividing by 255.
                single_patch_img = scaler.fit_transform(
                    single_patch_img.reshape(-1, single_patch_img.shape[-1])
                ).reshape(single_patch_img.shape)
                single_patch_img = np.expand_dims(single_patch_img, axis=0)
                pred = model.predict(single_patch_img)
                pred = np.argmax(pred, axis=3)
                pred = pred[0, :, :]

                patched_prediction.append(pred)

        patched_prediction = np.array(patched_prediction)
        patched_prediction = np.reshape(
            patched_prediction,
            [
                patches_img.shape[0],
                patches_img.shape[1],
                patches_img.shape[2],
                patches_img.shape[3],
            ],
        )

        unpatched_prediction = unpatchify(
            patched_prediction, (large_img.shape[0], large_img.shape[1])
        )
        #  prediction_without_smooth_blending = self.label_to_rgb(unpatched_prediction, input_labels)
        
        ONLY A SANITY CHECK FOR US
        
        plt.figure(figsize=(12, 12))
        plt.imshow(prediction_without_smooth_blending)
        plt.axis("off")
        """

        ###################################################################################
        # Predict using smooth blending

        input_img = scaler.fit_transform(img.reshape(-1, img.shape[-1])).reshape(
            img.shape
        )

        # Use the algorithm. The `pred_func` is passed and will process all the image 8-fold by tiling small patches with overlap, called once with all those image as a batch outer dimension.
        # Note that model.predict(...) accepts a 4D tensor of shape (batch, x, y, nb_channels), such as a Keras model.
        predictions_smooth = predict_img_with_smooth_windowing(
            input_img,
            window_size=patch_size,
            subdivisions=2,  # Minimal amount of overlap for windowing. Must be an even number.
            nb_classes=n_classes,
            pred_func=(lambda img_batch_subdiv: model.predict((img_batch_subdiv))),
        )

        final_prediction = np.argmax(predictions_smooth, axis=2)

        ########################
        # Plot and save results
        prediction_with_smooth_blending = self.label_to_rgb(
            final_prediction, input_labels
        )
        #  prediction_without_smooth_blending = self.label_to_rgb(unpatched_prediction, input_labels)

        """
        SANITY CHECK ONLY
        plt.figure(figsize=(20, 20))
        plt.subplot(221)
        plt.title("Testing Image")
        plt.imshow(img)
        plt.subplot(222)

        plt.title("Prediction with smooth blending")
        plt.imshow(prediction_with_smooth_blending)
        """
        # plt.show() !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # Result image is shown
        # save_png_as_shp(
        #    prediction_with_smooth_blending,
        #    [label.RGBColor for label in input_labels],
        #    f"Results/{dataP[:-4]}",
        # )  # the list is empty for now but the identification colors should be passed here

        os.makedirs(f"Results/{dataP[:-4]}/", exist_ok=True)
        img_to_write = cv2.cvtColor(prediction_with_smooth_blending,  cv2.COLOR_RGB2BGR)
        cv2.imwrite(f"Results/{dataP[:-4]}/result.png", img_to_write)  # we write with cv2. the cv2 default is BGR so we have to switch back to BGR
        del(img_to_write)

        # TODO: Add epsg extraction from the picture
        crs = None
        extent = None
        with rasterio.open(dataP) as src:
            # Get the spatial reference information
            crs = src.crs
            extent = src.bounds
            x_resolution = src.res[0]
            y_resolution = src.res[1]

        new_id_index = save_as_single_shapefile(
            prediction_with_smooth_blending,
            [label.RGBColor for label in input_labels],
            crs,
            extent,
            x_resolution,
            y_resolution,
            id_counter,
            f"Results/{dataP[:-4]}",
            is_POI=self.IsPOI

        )
        return new_id_index

    def label_to_rgb(self, predicted_image, input_labels):
        segmented_img = np.empty(
            (predicted_image.shape[0], predicted_image.shape[1], 3)
        )
        for i in range(len(input_labels)):
            segmented_img[(predicted_image == i)] = input_labels[i].RGBColor
        # segmented_img[(predicted_image == 0)] = Building
        # segmented_img[(predicted_image == 1)] = Land
        # segmented_img[(predicted_image == 2)] = Road
        # segmented_img[(predicted_image == 3)] = Vegetation
        # segmented_img[(predicted_image == 4)] = Water
        # segmented_img[(predicted_image == 5)] = Unlabeled

        segmented_img = segmented_img.astype(np.uint8)
        return segmented_img

    ###################
    # Convert labeled images back to original RGB colored masks.

    """
    def label_to_rgb(self, predicted_image):

        Building = '#3C1098'.lstrip('#')
        Building = np.array(tuple(int(Building[i:i+2], 16) for i in (0, 2, 4))) # 60, 16, 152

        Land = '#8429F6'.lstrip('#')
        Land = np.array(tuple(int(Land[i:i+2], 16) for i in (0, 2, 4))) #132, 41, 246

        Road = '#6EC1E4'.lstrip('#')
        Road = np.array(tuple(int(Road[i:i+2], 16) for i in (0, 2, 4))) #110, 193, 228

        Vegetation =  'FEDD3A'.lstrip('#')
        Vegetation = np.array(tuple(int(Vegetation[i:i+2], 16) for i in (0, 2, 4))) #254, 221, 58

        Water = 'E2A929'.lstrip('#')
        Water = np.array(tuple(int(Water[i:i+2], 16) for i in (0, 2, 4))) #226, 169, 41

        Unlabeled = '#9B9B9B'.lstrip('#')
        Unlabeled = np.array(tuple(int(Unlabeled[i:i+2], 16) for i in (0, 2, 4))) #155, 155, 155



        segmented_img = np.empty((predicted_image.shape[0], predicted_image.shape[1], 3))

        segmented_img[(predicted_image == 0)] = Building
        segmented_img[(predicted_image == 1)] = Land
        segmented_img[(predicted_image == 2)] = Road
        segmented_img[(predicted_image == 3)] = Vegetation
        segmented_img[(predicted_image == 4)] = Water
        segmented_img[(predicted_image == 5)] = Unlabeled

        segmented_img = segmented_img.astype(np.uint8)
        return(segmented_img)
        """

    """
    #plt.title('Testing Label')
    #plt.imshow(original_mask)
    #plt.subplot(223)
    plt.title('Prediction without smooth blending')
    plt.imshow(prediction_without_smooth_blending)
    plt.subplot(224)
    """

    #############################
