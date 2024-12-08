import os

import cv2
import numpy as np
import rasterio
from rasterio.windows import Window
from joblib import Parallel, delayed
from tqdm import tqdm
from tqdm_joblib import tqdm_joblib
from rasterio.enums import Resampling
#from osgeo import gdal
from skimage.segmentation import slic
from skimage.color import label2rgb

from training_data_maker import shp_maker as get_shp
from training_data_populator import training_data_populator as pop
from small_pic_inMemory_DTO import small_tif_data_container


class Converter:
    def __init__(self, inputDir: str, outputDir: str, splitSize, input_labels,
                 gausKernel: int = 7, gausSigma:float = 0, clipLimit: float = 0.5, tileGridSize: int = 20,
                 marker_size: int = 16, is_inmemory: bool = False, label_balance: float = 0.1, use_preprocess: bool = True):
        self.input = inputDir
        self.output = outputDir
        self.split_size = splitSize
        self.Output = []
        self.labels = input_labels

        self.gausKernel = gausKernel
        self.gausSigma = gausSigma
        self.clipLimit = clipLimit
        self.tileGridSize = tileGridSize
        self.marker_size = marker_size
        self.populator = pop(self.output, label_balance)
        self.is_inmemory = is_inmemory
        self.is_use_preprocess = use_preprocess


    def image_dimension_modifier(self, file_path, file_name, target_resolution_cm):
        print(f"{file_name} dimensions being checked")
        with rasterio.open(f"{file_path}/{file_name}") as src:
            transform = src.transform
            src_res_x = np.round(transform.a, 3)
            src_res_y = np.round(np.abs(transform.e), 3)

            if abs(src_res_x) != abs(src_res_y):
                raise ValueError("Pixels are not square")

            del src_res_y  # we don't need this since this is the same as x
            src_res_cm = src_res_x * 100

            if src_res_cm == target_resolution_cm:
                print(f"The {file_name} file was in the right size.")
                return file_path

            scalar = src_res_cm / target_resolution_cm
            dst_width = int(src.width * scalar)
            dst_height = int(src.height * scalar)

            data = src.read(
                out_shape=(
                    src.count,
                    dst_height,
                    dst_width
                ),
                resampling=Resampling.cubic
            )

            new_transform = transform * transform.scale(
                (src.width / dst_width),
                (src.height / dst_height)
            )

            dst_meta = src.meta.copy()
            dst_meta.update({
                'height': dst_height,
                'width': dst_width,
                'transform': new_transform
            })

            new_file_name = f"{file_name.split('.')[-2]}_resampled.tif"
            with rasterio.open(f"{file_path}/{new_file_name}", 'w', **dst_meta) as dst:
                dst.write(data)

            return new_file_name

    def convert(self,convert_file_name: str = None, isTraining: bool = False, isPopulate:bool = False, isUseBackGround:bool = False, ):
        if convert_file_name and self.is_inmemory:
            tif_in = f"{convert_file_name}"
            tif_out = f"{self.output}"
            os.makedirs(tif_out, exist_ok=True)
            self.split_large_tif(tif_in, tif_out, '0', isTraining, isPopulate, isUseBackGround)
        else:
            for path, subdirs, files in os.walk(self.input):

                images = os.listdir(path)  # List of all image names in this subdirectory
                for i, image_name in enumerate(images):
                    if image_name.endswith(".tif"):  # Only read jpg images...
                        #img_name = self.image_dimension_modifier(path, image_name, 1.00)  #TODO clean up after ourselves

                        tif_in = f"{path}/{image_name}"
                        tif_out = f"{self.output}/images"
                        os.makedirs(tif_out, exist_ok=True)
                        self.split_large_tif(tif_in, tif_out, str(i), isTraining, isPopulate, isUseBackGround)

        if isTraining:
            self.populator.populate_images(isPopulate, isUseBackGround)
        return self.Output  # image path str-s or array of np.array

    def split_large_tif(self, input_tif, output_folder, id, isTraining, isPopulate, isUseBackGround):
        # Open the large GeoTIFF file
        with rasterio.open(input_tif) as src:
            # Get the dimensions of the large image
            height, width = src.height, src.width

        overlap_percent = 0.0  # use only on smallest of images. Make it toggleable

        stride_x = int(round(self.split_size[0] * (1 - overlap_percent)))
        stride_y = int(round(self.split_size[1] * (1 - overlap_percent)))

        stride_x = max(stride_x, 1)
        stride_y = max(stride_y, 1)

        # Get the number of splits in each direction
        num_splits_x = int(np.ceil((width - self.split_size[0]) / stride_x)) + 1
        num_splits_y = int(np.ceil((height - self.split_size[1]) / stride_y)) + 1
        # Get the leftover pixels in case the image dimensions are not divisible by the split size
        self.leftover_x = (stride_x * (num_splits_x - 1) + self.split_size[0]) - width
        self.leftover_y = (stride_y * (num_splits_y - 1) + self.split_size[1]) - height

        indices = [(i, j) for i in range(int(num_splits_y)) for j in range(int(num_splits_x))]

        with tqdm_joblib(tqdm(desc="Processing", total=len(indices))) as progress_bar:
            results = Parallel(n_jobs=6)(
                delayed(self.process_split)(
                    i, j, input_tif, output_folder, id, isTraining, isPopulate, isUseBackGround, stride_x, stride_y
                ) for i, j in indices
            )

        for converted_image_data, is_bad in results:
            if converted_image_data is not None:
                if is_bad:
                    self.populator.save_image(converted_image_data)
                self.Output.append(converted_image_data)


    def process_split(self, i, j, input_tif, output_folder, id, isTraining, isPopulate, isUseBackGround, stride_x, stride_y):
        with rasterio.open(input_tif) as src:
            x_resolution = src.res[0]
            y_resolution = src.res[1]

            x_offset = j * stride_x  # j * (self.split_size[0] * x) where x is a float  0 < x <= 1 && x is the amount of stride as a percentage, For starter I would use 0.25
            y_offset = i * stride_y # i * (self.split_size[1] * x) -''-
            width_crop = self.split_size[0]
            height_crop = self.split_size[1]

            # Adjust the cropping window for the right and bottom edges (accounting for leftovers)
            if x_offset + width_crop > src.width:
                x_offset = src.width - width_crop
                x_offset = max(x_offset, 0)  # Ensure offset is not negative
            if y_offset + height_crop > src.height:
                y_offset = src.height - height_crop
                y_offset = max(y_offset, 0)  # Ensure offset is not negative

            # Read the split image data
            window = Window(x_offset, y_offset, width_crop, height_crop)
            data = src.read(window=window)

            # Convert to RGB cv2
            img = np.transpose(data, (1, 2, 0))
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
            image_original = np.copy(img)

            if self.is_use_preprocess:
                ## NOISE REDUCTION
                # Apply gaussian blur
                img = cv2.GaussianBlur(img, (3, 3), 0) # nov 4 óta van bent

                # ## CLAHE
                #img = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)

                # Separate the image parts
                # l_channel, a, b = cv2.split(img)
                #
                # # Apply Contrast Limited Adaptive Histogram Equalization
                # clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(3, 3))
                # l_channel = clahe.apply(l_channel)
                #
                # # merge the parts back
                # img = cv2.merge((l_channel, a, b))
                #
                # # convert back into RBGA
                # img = cv2.cvtColor(img, cv2.COLOR_LAB2BGR)

                # # SLIC
                # segments = slic(img, 1500, 10)
                # img = label2rgb(segments, img, kind='avg')
                #
                # # OTSU
                # gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                #
                # _, mask = cv2.threshold(
                #     gray_img, 100, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU,
                # )
                #
                # img = cv2.bitwise_and(img, img, mask=mask)

            data = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)

            # Create a new output filename
            output_filename = f"{output_folder}/split{id}_{i}_{j}.tif"



            pixel_count = width_crop * height_crop
            has_enought_brown_pixels = self._has_brown_patches(data, pixel_count, percentage=0.01)
            #has_enought_brown_pixels = True

            data = np.transpose(data, (2, 0, 1))  # this has to be here this order is only good for the rasterio type images but not for anything else
            if has_enought_brown_pixels or isTraining:
                # Write the split image data to the output file
                with rasterio.open(
                        output_filename,
                        "w",
                        driver="GTiff",
                        width=width_crop,
                        height=height_crop,
                        count=src.count,
                        dtype=src.dtypes[0],
                        crs=src.crs,
                        transform=src.window_transform(window),
                ) as dst:
                    try:
                        dst.write(data)
                        print(f"Split{id} {i}_{j} write succeeded.")
                    except Exception as e:
                        print(f"Split{id} {i}_{j} write failed. Error: {str(e)}")

            if isTraining:
                os.makedirs(self.output + "/masks", exist_ok=True)
                os.makedirs(self.output + "/binLabels", exist_ok=True)
                os.makedirs(self.output + "/labels", exist_ok=True)
                os.makedirs(self.output + "/cropped", exist_ok=True)

                shp_out = f"{self.output}/masks/split{id}_{i}_{j}_mask.png"
                shp_folder = os.path.join(self.input, "shapes")

                shp_getter = get_shp(
                    shp_folder, shp_out, output_filename, self.labels,
                    x_resolution, y_resolution, marker_size=self.marker_size,
                    image_original=image_original, temp_folder=self.output, cropped_shp_id=f"split{id}_{i}_{j}"
                )

                shp_getter.crop_shapefile_with_geotiff()
                was_successful, is_flawed = shp_getter.shapefile_to_colored_png()
                # is_flawed: seed point was inadaquate
                # was_successful: we generated boxes.

                # if was_successful:
                #     self.populator.image_with_label_count += 1

                if not was_successful and not is_flawed and (isPopulate or isUseBackGround):
                    _, width, height = data.shape
                # no points were found and this is not because the seed point was wrong.

                    if not self.is_inmemory and has_enought_brown_pixels:
                        with rasterio.open(
                                output_filename,
                                "w",
                                driver="GTiff",
                                width=width_crop,
                                height=height_crop,
                                count=src.count,
                                dtype=src.dtypes[0],
                                crs=src.crs,
                                transform=src.window_transform(window),
                        ) as dst:
                            try:
                                dst.write(data)
                                print(f"Split{id} {i}_{j} write succeeded.")
                            except Exception as e:
                                print(f"Split{id} {i}_{j} write failed. Error: {str(e)}")
                        #self.populator.save_image(output_filename)

            if self.is_inmemory:
                new_bound = rasterio.windows.bounds(window, src.transform)

                new_bounds = {
                    "left": new_bound[0],
                    "right": new_bound[2],

                    "top": new_bound[3],
                    "bottom": new_bound[1],
                }

                return small_tif_data_container(data, src.crs, bounds=src.bounds, res_x=src.res[0], res_y=src.res[1], x=j, y=i)
            else:
                return output_filename, (isTraining and not was_successful and not is_flawed and (isPopulate or isUseBackGround) and not self.is_inmemory and has_enought_brown_pixels)

    def _has_brown_patches(self, image, pixel_number, percentage=0.10):
        # COLOR_BGR2RGBA
        rgb_image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        hsv_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)

        # h 10 - 35
        # s 20 - 60
        # v 50 - 100
        lower_brown = np.array([5, 51, 127])
        higher_brown = np.array([17, 153, 255])

        mask = cv2.inRange(hsv_image, lower_brown, higher_brown)

        brown_pixel_count = cv2.countNonZero(mask)

        threshold = pixel_number * percentage

        return brown_pixel_count > threshold


