import random
import os

import cv2
import numpy as np
import rasterio
from ultralytics.data.utils import autosplit

from training_data_maker import shp_maker


class training_data_populator:
    def __init__(self, main_path, background_label_balance:float=0.25, synthetic_image_balance: float = 0.28):
        self.main_path = main_path
        self.label_path = main_path + "/labels"
        self.image_path = main_path + "/images"

        self.images = []
        self.original_image_count = 0
        self.image_with_background_count = 0
        self.image_with_synthetic_data_count = 0

        self.background_label_balance = background_label_balance
        self.synthetic_image_balance = synthetic_image_balance  # Based on KPi covid data 85 - 95 peak


    def save_image(self, file_name):
        self.images.append(file_name)

    def populate_images(self, isPopulate, isUseBackGround):
        autosplit(self.image_path, weights=(0.9, 0.1, 0.0), annotated_only=True)
        if not isPopulate and not isUseBackGround:
            return


        with open(self.main_path+"\\autosplit_train.txt") as f:
            files = [line.strip().split("/")[-1]for line in f]

        # calculate the annotated count(files.len)
        self.original_image_count = len(files)

        with open(self.main_path+"\\autosplit_val.txt") as f:
            val_files = [line.strip().split("/")[-1]for line in f]


        validation_image_count = len(val_files)
        validation_background_images = 0

        del val_files

        random.seed(42)
        random.shuffle(self.images)

        for j, image in enumerate(self.images):
            #check if this image is mostly white (more than 75%)
            if self.is_image_mostly_white(image):
                continue

            shouldAddSyntheticImage = self.check_synthetic_label_data()

            # Case for adding an annotated image
            if shouldAddSyntheticImage and isPopulate:
                _ = self.add_populated_image_to_dataset(files, image, j)
                self.image_with_synthetic_data_count += 1

                # ADD TO THE autosplit_train.txt the result value
                # image.split('/')[-1][:-4] check this in the labels.txt folder. If exists there add to the autosplit.
                label_name = image.split('/')[-1][:-4]
                self.add_new_data_to_train_file(label_name)
                continue

            shouldAddBackgroundImage = self.check_backround_label_balance()

            # add case for adding as unannotated image
            if shouldAddBackgroundImage and isUseBackGround:
                # no labels text
                shouldAddBackgroundToVal = (validation_background_images / validation_image_count) < self.background_label_balance
                if shouldAddBackgroundToVal:
                    validation_background_images += 1
                else:
                    self.image_with_background_count += 1

                label_name = image.split('/')[-1][:-4] + ".txt"

                with open(os.path.join(self.label_path, label_name), 'w'):
                    pass

                # ADD TO THE autosplit_train.txt the result value
                # image.split('/')[-1][:-4] check this in the labels.txt folder. If exists there add to the autosplit.
                label_name = image.split('/')[-1][:-4]
                self.add_new_data_to_train_file(label_name, shouldAddBackgroundToVal)

        with open("data distribution.txt", "w") as f:
            f.write(f"original amount: {self.original_image_count}\r\nSynthetic data: {self.image_with_synthetic_data_count}\r\nbackground amount: {self.image_with_background_count}")

        print("original amount: " + str(self.original_image_count))
        print("Synthetic data: " + str(self.image_with_synthetic_data_count))
        print("background amount: " + str(self.image_with_background_count))



    def add_populated_image_to_dataset(self, files, image, j):
        add_burrow_num = 1 #random.randint(1, 2) random doesn't needed because the images are small
        for i in range(add_burrow_num):
            with rasterio.open(image) as src:
                img = src.read()  # channel, height, width
                img = np.transpose(img, (1, 2, 0))  # height, width,  channel
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)

            is_appended = False
            retry_limit = 10
            while not is_appended and retry_limit > 0:
                region, x1, y1, x2, y2, center, width, height = self._get_object(files)

                # region and image both height, width, channel
                try:
                    img[y1:y2, x1:x2] = region
                except Exception as e:
                    print(e)
                    print("region was flawed")
                    retry_limit -= 1
                    continue

                is_appended = self.append_txt(img_shape_info=img.shape, class_key=0, center=center, width=width,
                                              height=height, path=image)
                retry_limit -= 1

            with rasterio.open(
                    image,
                    "w",
                    driver="GTiff",
                    count=src.count,
                    width=src.width,
                    height=src.height,
                    dtype=src.dtypes[0],
                    crs=src.crs
            ) as dst:
                try:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)
                    dst.write(np.transpose(img, (2, 0, 1)))
                    print(f"image at {image} is redeemed.")

                except Exception as e:
                    print(f"Split{id} {i}_{j} write failed. Error: {str(e)}")
        return img

    def _get_object(self, files):
        #files = os.listdir(self.label_path)

        # Filter out directories, only keep files which are part of the test cases

        #files = [f for f in files if os.path.isfile(os.path.join(self.label_path, f))]

        if not files:
            raise FileNotFoundError("No files found in the directory.")
        image = None
        while True:
            try:
                # Randomly select a file from the list
                random_file = random.choice(files)[:-4]
                tif_file = random_file + ".tif"
                txt_file = random_file + ".txt"

                opened_file = open(os.path.join(self.label_path, txt_file), 'r')
                lines = [line for line in opened_file if line.strip() != '']
                opened_file.close()
                data = random.choice(lines).strip()
                if data == '':
                    raise ValueError("data is poorly choosen")

                # Split the chosen data block into individual elements
                data_elements = data.split()

                data = [float(element) for element in data_elements]

                ## Open original image
                image = cv2.imread(f"{self.image_path}/{tif_file}")
                if image is None:
                    raise ValueError("Image is None")

                break
            except:
                print(f"Error!. Retrying...")
                continue

        height, width, _ = image.shape  # height, width, channels
        _, center_x, center_y, img_width, img_height = data
        center_x *= width
        img_width *= width

        center_y *= height
        img_height *= height

        # extract image
        # Calculate the top-left corner of the region
        puff = 15 # puffer to copy the sourroundings of the burrow not just that solely
        x1 = int(max(0, center_x - (img_width + puff) // 2))
        y1 = int(max(0, center_y - (img_height + puff) // 2))

        # Calculate the bottom-right corner of the region
        x2 = int(min(width, center_x + (img_width + puff) // 2))
        y2 = int(min(height, center_y + (img_height + puff) // 2))

        # Extract and return the region
        region = image[y1:y2, x1:x2]  # height and width
        # cv2.imshow("original", region)
        # cv2.waitKey()

        return region, x1, y1, x2, y2, (center_x, center_y), img_width, img_height

    def append_txt(self, img_shape_info, class_key, center, width, height, path):
        img_width = img_shape_info[1]
        img_height = img_shape_info[0]

        c_x = int(center[0])
        c_y = int(center[1])


        new_text = f"{class_key} {c_x/img_width} {c_y/img_height} {width/img_width} {height/img_height}"

        is_new = not self.is_line_already_exists(path, new_text)

        if is_new:
            file_name = path.split('/')[-1][:-4]

            try:
                file_has_lines = False
                with open(f"tmp/labels/{file_name}.txt", "r") as file:
                    if file.readlines():
                        file_has_lines = True
            except FileNotFoundError:
                file_has_lines = False

            f = open(f"tmp/labels/{file_name}.txt", 'a')
            if file_has_lines:
                f.write("\r\n")
            f.write(new_text)
            f.close()
            return True
        else:
            return False

    def is_line_already_exists(self, path, text):
        file_name = path.split('/')[-1][:-4]
        if os.path.isfile(f"tmp/labels/{file_name}.txt"):
            f = open(f"tmp/labels/{file_name}.txt", 'r')
            lines = f.read().split("\n")

            comp = text.strip()
            for line in lines:
                if line == comp:
                    return True

        else:
            return False

        return False

    def add_new_data_to_train_file(self, label_name, isToValidation:bool = False):
        if os.path.exists(os.path.join(self.label_path, label_name + ".txt")):
            if isToValidation:
                with open(self.main_path + "\\autosplit_val.txt", 'a') as f:
                    f.write(f"./images/{label_name}.tif\r\n")
            else:
                with open(self.main_path + "\\autosplit_train.txt", 'a') as f:
                    f.write(f"./images/{label_name}.tif\r\n")

    def check_backround_label_balance(self):
        balance = self.image_with_background_count / (self.original_image_count + self.image_with_synthetic_data_count)  # > 0 if we have more unannotated image than annotated

        return balance < self.background_label_balance

    def check_synthetic_label_data(self):
        balance = self.image_with_synthetic_data_count / self.original_image_count  # > 0 if we have synthethic data

        return balance < self.synthetic_image_balance

    def is_image_mostly_white(self, image, white_threshold=0.75, white_pixel_threshold=200):
        with rasterio.open(image) as src:
            img = src.read()
            img = np.transpose(img, (2, 1, 0))
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)


        # Check if pixels are white (all channels are close to 255)
        is_white = np.all(img >= white_pixel_threshold, axis=2)
        is_black = np.all(img <= 55, axis=2)

        # Calculate the percentage of white pixels
        white_pixel_percentage = np.mean(is_white)
        black_pixel_percentage = np.mean(is_black)

        return white_pixel_percentage >= white_threshold or black_pixel_percentage >= white_threshold







# pop = training_data_populator("G:\\SULI\\Projectmunka\\git\\NeuralNetworkForFoliageDetection\\tmp")
#
# pop.save_image("G:\SULI\Projectmunka\git\\NeuralNetworkForFoliageDetection\\tmp\images\split0_1_25.tif")
#
# pop.populate_images()



