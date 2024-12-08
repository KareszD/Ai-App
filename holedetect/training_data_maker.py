import os
import math

import cv2
import numpy as np
import shapely
#from osgeo import ogr, gdal
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import rasterio
from rasterio.mask import mask
from PIL import Image, ImageDraw
from rasterio.crs import CRS
from shapely.geometry import Polygon
from geopandas.tools import overlay
from skimage.measure import regionprops
import pngcanvas

class shp_maker:
    def __init__(self, shapefile_path, output_image_path, geotiff_path, labels, x_resolution, y_resolution, marker_size,
                 image_original, cropped_shp_id, threshold_value: int = 25, temp_folder:str = "tmp"):
        self.shapefile_path = shapefile_path  # "2022_10_27-Mész-tető_HRAMN_IS.shp"
        self.output_image_path = output_image_path  # "output_image.png"  # Replace this with the desired output PNG image path
        self.geotiff_path = geotiff_path  # "meszteto.tif"
        self.x_resolution = x_resolution
        self.y_resolution = y_resolution

        # Constants
        self.croped_shapefile = f"{temp_folder}\\cropped\\{cropped_shp_id}_croped.shp"
        self.column_name = "LC"
        self.marker_size = marker_size
        self.is_flawed = False

        self.original_image = image_original
        self.bin_img = None
        self.is_image_preprocessed = False
        self.threshold_ori = threshold_value
        self.threshold = self.threshold_ori
        self.temp = temp_folder


        # Define the color mapping (replace values and colors as per your requirements)


        self.color_mapping = {label.ID: label.HexaColor for label in labels}

    def crop_shapefile_with_geotiff(self):
        # Read the shapefile into a GeoDataFrame
        gdf = gpd.read_file(self.shapefile_path)
        if gdf.crs is None:
            # Set the CRS using EPSG code (example: EPSG 4326 for WGS 84)
            gdf.crs = 'EPSG:23700'

        # Read the GeoTIFF file
        with rasterio.open(self.geotiff_path, 'r+') as src:
            if src.crs is None:
                src.crs = CRS.from_epsg(23700)

                # Get the extent (bounding box) of the GeoTIFF
            self.tiff_extent = src.bounds

            # Create a bounding box polygon from the extent of the GeoTIFF
            tiff_polygon = Polygon([
                (self.tiff_extent.left, self.tiff_extent.bottom),
                (self.tiff_extent.left, self.tiff_extent.top),
                (self.tiff_extent.right, self.tiff_extent.top),
                (self.tiff_extent.right, self.tiff_extent.bottom)
            ])

            # Perform a spatial overlay to retain intersecting features and attributes
            gdf_cropped = overlay(gdf, gpd.GeoDataFrame({'geometry': [tiff_polygon]}, crs=src.crs),
                                  how='intersection')
            # Compute the intersection of each polygon with the bounding box of the GeoTIFF
            # #gdf_cropped = gdf.intersection(tiff_polygon)

            # Remove any empty geometries resulting from the intersection
            #gdf_cropped = gdf_cropped[~gdf_cropped.is_empty]
            # Get the extent (bounding box) of the GeoTIFF
            #tiff_extent = src.bounds

            # Perform spatial intersection to get only the intersecting polygons
            #gdf_cropped = gdf.cx[tiff_extent[0]:tiff_extent[2], tiff_extent[1]:tiff_extent[3]]

        # Save the cropped GeoDataFrame to a new shapefile
        gdf_cropped.to_file(self.croped_shapefile)

        # Get the width and height of the GeoTIFF
        tiff_width = src.width
        tiff_height = src.height

        # Calculate the aspect ratio of the GeoTIFF
        aspect_ratio = tiff_width / tiff_height

        # Specify the desired PNG width (adjust as needed)
        self.png_width = tiff_width

        # Calculate the corresponding PNG height based on the aspect ratio
        self.png_height = int(self.png_width / aspect_ratio)

    def draw_polygon(self, draw, polygon, minx, miny, color, scale_factor_x, scale_factor_y):
        # Convert the geometry to pixel coordinates relative to the canvas
        x, y = polygon.exterior.coords.xy
        x_pixels = [(xi - minx) * scale_factor_x for xi in x]
        y_pixels = [(yi - miny) * scale_factor_y for yi in y]
        coords = list(zip(x_pixels, y_pixels))

        # Draw the polygon on the image using the specified color
        draw.polygon(coords, fill=color)

    def shapefile_to_colored_png(self):
        # Read the shapefile into a GeoDataFrame
        gdf = gpd.read_file(self.croped_shapefile)

        # Create a custom colormap based on the color_mapping dictionary
        cmap = ListedColormap([self.color_mapping[val] for val in sorted(self.color_mapping.keys())])

        # Get the bounding box (extent) of the shapefile
        minx, miny, maxx, maxy = gdf.total_bounds  # ideally the same to the tiff_extent
        if (minx == self.tiff_extent.left and miny == self.tiff_extent.bottom and maxx == self.tiff_extent.right and maxy == self.tiff_extent.top):


            scale_factor_x = self.png_width / (maxx - minx)
            scale_factor_y = self.png_height / (maxy - miny)
        elif not gdf.empty:
            print("BOUNDING BOX SIZE MISMATCH! Are you trying to parse a collection of points?")
            print(str(maxx - minx))
            print(str(maxy - miny))
            minx = self.tiff_extent.left
            miny = self.tiff_extent.bottom
            maxx = self.tiff_extent.right
            maxy = self.tiff_extent.top

            scale_factor_x = self.png_width / (maxx - minx)
            scale_factor_y = self.png_height / (maxy - miny)
        elif gdf.empty:
            scale_factor_x = 0
            scale_factor_y = 0

        # Calculate the width and height of the canvas
        canvas_width = self.safe_int_conversion((maxx - minx) * scale_factor_x)
        canvas_height = self.safe_int_conversion((maxy - miny) * scale_factor_y)

        if canvas_width < 640 and canvas_width > 0:
            print("BAJ VAN!")

        # Create a new blank image with a white background
        img = Image.new('RGB', (canvas_width, canvas_height), color='black')
        draw = ImageDraw.Draw(img)

        # Draw the shapefile polygons on the canvas
        for geom in gdf['geometry']:
            if geom.geom_type == 'Polygon':
            # Convert the geometry to pixel coordinates relative to the canvas
            #x, y = geom.exterior.coords.xy
            #x_pixels = [(xi - minx) for xi in x]
            #y_pixels = [(yi - miny) for yi in y]
            #coords = list(zip(x_pixels, y_pixels))

            # Draw the polygon on the canvas using the specified color
                key = gdf.loc[gdf['geometry'] == geom, self.column_name].values[0]
                if self.column_name and not math.isnan(key):
                    # Replace 'column_value' with the column value you want to use for coloring
                    color = self.color_mapping[key]  # cmap.get(gdf.loc[gdf['geometry'] == geom, self.column_name].values[0], '#000000')
                else:
                    color = '#FFFFFF'  # Default color if no self.column_name is provided
                self.draw_polygon(draw, geom, minx, miny, color, scale_factor_x, scale_factor_y)
            elif geom.geom_type == 'MultiPolygon':
                for polygon in geom.geoms:
                    key = gdf.loc[gdf['geometry'] == geom, self.column_name].values[0]
                    if self.column_name and not math.isnan(key):
                        # Replace 'column_value' with the column value you want to use for coloring
                        color = self.color_mapping[key]  # cmap.get(gdf.loc[gdf['geometry'] == geom, self.column_name].values[0], '#000000')
                    else:
                        color = '#FFFFFF'  # Default color if no self.column_name is provided
                    self.draw_polygon(draw, polygon, minx, miny, color, scale_factor_x, scale_factor_y)
            elif geom.geom_type == "Point":
                key = 0
                color = self.color_mapping[key]

                polygon, self.is_flawed = self.create_polygon_from_point(geom)
                if self.is_flawed:
                    break

                self.draw_polygon(draw, polygon, minx, miny, color, scale_factor_x, scale_factor_y)

            #draw.polygon(coords, fill=color)

        try:
            # Save the image as a PNG file
            img = img.transpose(Image.ROTATE_180)
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            width, height = img.size
            if self.is_flawed:
                print("Seed point wrongly given")
                if os.path.exists(self.geotiff_path):
                    os.remove(self.geotiff_path)
                    print(f"File '{self.geotiff_path}' has been deleted.")
                    file_name = self.geotiff_path.split('/')[-1][:-4]
                    if os.path.isfile(f"{self.temp}/labels/{file_name}.txt"):
                        os.remove(f"{self.temp}/labels/{file_name}.txt")
                    return False, self.is_flawed
            elif (width ==0 or height == 0):
                file_name = self.geotiff_path.split('/')[-1][:-4]
                if os.path.isfile(f"{self.temp}/labels/{file_name}.txt"):
                    #img.save(self.output_image_path)
                    print("Only txt is added to the training files")
                    return True, self.is_flawed
                else:
                    print("Incorrect size of mask generated, the corresponding training image get's deleted")  #TODO Miért van ebből ilyen sok?
                    if os.path.exists(self.geotiff_path):
                        os.remove(self.geotiff_path)
                        print(f"File '{self.geotiff_path}' has been deleted.")
                        return False, self.is_flawed
            else:
                img.save(self.output_image_path)
                print("Image saved successfully.")
                return True, self.is_flawed
        except Exception as e:
            print(f"Failed to save the image. Error: {str(e)}")
            if gdf.empty:
                print(f"there is no data for the image at {self.geotiff_path} the image got deleted from training")
                if os.path.exists(self.geotiff_path):
                    os.remove(self.geotiff_path)
                    print(f"File '{self.geotiff_path}' has been deleted.")

            return False, self.is_flawed
        plt.close()

    def create_polygon_from_point(self, geom: shapely.Point):
        if not self.is_image_preprocessed or self.bin_img is None:
            self.threshold = self.threshold_ori
            self._preprocess_original_image()
        still_growing = True
        while still_growing:
            # corresponding file path should be self. self.output_image_path
            center_point, width, height, is_flawed, still_growing = self.region_growing(geom)
        # area = self.marker_size  #16  # pixel
        x_skew = width/2
        y_skew = height/2

        point_xmin_ymax = (center_point[0] - x_skew, center_point[1] + y_skew)  # top left
        point_xmax_ymax = (center_point[0] + x_skew, center_point[1] + y_skew)  # top right
        point_xmax_ymin = (center_point[0] + x_skew, center_point[1] - y_skew)  # bottom right
        point_xmin_ymin = (center_point[0] - x_skew, center_point[1] - y_skew)  # bottom left


        return Polygon([point_xmin_ymax, point_xmax_ymax, point_xmax_ymin, point_xmin_ymin, point_xmin_ymax]), is_flawed


    def _preprocess_original_image(self):
        binary_image = np.zeros_like(self.original_image[:, :, 0])
        for i in range(self.original_image.shape[0]):
            for j in range(self.original_image.shape[1]):
                a, b, c = self.original_image[i, j].astype(np.int16)
                ab, ac, bc = np.abs(a - b), np.abs(a - c), np.abs(b - c)

                th = self.threshold
                if ab < th and bc < th and ac < th:
                    binary_image[i, j] = 255
                else:
                    binary_image[i, j] = 0

        self.bin_img = binary_image
        print(f"\033[93mThreshold value:{self.threshold}\033[0m")

        cv2.imwrite(f"{self.temp}/binLabels/{self.geotiff_path.split('/')[-1][:-4]}.png", binary_image)

        self.is_image_preprocessed = True

    def region_growing(self, seed_geom: shapely.Point):
        with rasterio.open(self.geotiff_path) as src:
            #img = src.read().astype(np.int32)
            transform = src.transform
            extent = src.bounds
            x_resolution = src.res[0]
            y_resolution = src.res[1]

        img = self.bin_img.astype(np.int32)

        mask = np.zeros_like(img, dtype=np.uint8)
        seed_img_dim = rasterio.transform.rowcol(transform, seed_geom.xy[0][0], seed_geom.xy[1][0])
        seed_point = (int(seed_img_dim[1]), int(seed_img_dim[0]))
        seed_value = img[seed_img_dim[0], seed_img_dim[1]]
        if seed_value == 0:
            if self.threshold < 55:
                self.threshold += 5
                self._preprocess_original_image()
                return (256, 256), 0, 0, True, True
            else:
                if self.threshold_ori != self.threshold:
                    self.threshold = self.threshold_ori
                    self._preprocess_original_image()
                return (256, 256), 0, 0, True, False


        # Region growing parameters
        tolerance = 0

        queue = [seed_point]
        visited = set()
        found_points = 0
        while queue:
            current_point = queue.pop(0)
            visited.add(current_point)
            if (0 <= current_point[0] < img.shape[1]) and (0 <= current_point[1] < img.shape[0]):
                current_value = img[current_point[1], current_point[0]]
                #
                # rgb_array = current_value[:3]
                # differences = np.abs(np.diff(np.sort(rgb_array)))
                # max_diff = np.max(differences)
                # if max_diff <= 40 or np.all(np.abs(current_value - seed_value) == 0):
                abs_value = np.abs(current_value - seed_value)
                is_smaller = abs_value <= tolerance
                if np.all(is_smaller): # optimize it
                    mask[current_point[1], current_point[0]] = 255
                    found_points += 1
                    if found_points >= 7225:  # 85*85
                        break

                    # Add neighbours
                    neighbors = [(current_point[0] + i, current_point[1] + j) for i in [-1, 0, 1] for j in [-1, 0, 1]]
                    for neighbor in neighbors:
                        if neighbor not in queue:
                            if neighbor not in visited:
                                queue.append(neighbor)


        region_prop = regionprops(mask)
        center_point_img = (region_prop[0].centroid)
        print(center_point_img)
        center_point_geo = self.transform_coordinates(center_point_img[1], center_point_img[0], transform)

        # Find widest width and highest height
        width_img = region_prop[0].bbox[3] - region_prop[0].bbox[1]
        height_img = region_prop[0].bbox[2] - region_prop[0].bbox[0]
        width_geo = width_img * self.x_resolution
        height_geo = height_img * self.y_resolution

        if width_img > 80 or height_img > 80:
            if self.threshold_ori != self.threshold:
                self.threshold = self.threshold_ori
                self._preprocess_original_image()
            return (256, 256), 0, 0, True, False

        self.append_txt(img.shape, class_key=0, center=center_point_img, width=width_img, height=height_img)

        if self.threshold_ori != self.threshold:
            self.threshold = self.threshold_ori
            self._preprocess_original_image()

        return center_point_geo, width_geo, height_geo, False, False

    def transform_coordinates(self, x, y, transform):
        # Convert image coordinates to georeferenced coordinates
        # x_geo = extent.left + x * self.x_resolution
        # y_geo = extent.bottom + y * self.y_resolution

        x_geo, y_geo = rasterio.transform.xy(transform, y, x)
        return x_geo, y_geo

    def append_txt(self, img_shape_info, class_key, center, width, height):

        file_name = self.geotiff_path.split('/')[-1][:-4]
        try:
            file_has_lines = False
            with open(f"{self.temp}/labels/{file_name}.txt", "r") as file:
                if file.readlines():
                    file_has_lines = True
        except FileNotFoundError:
            file_has_lines = False


        f = open(f"{self.temp}/labels/{file_name}.txt", "a")
        if file_has_lines:
            f.write("\n")

        img_width = img_shape_info[-1]
        img_height = img_shape_info[-2]

        c_x = int(center[1])
        c_y = int(center[0])

        point = (c_x/img_width, c_y/img_height)


        f.write(f"{class_key} {c_x/img_width} {c_y/img_height} {width/img_width} {height/img_height}")
        f.close()


    def safe_int_conversion(self, value):
        try:
            if not math.isnan(value):
                return int(value)
        except TypeError:
            pass  # Handle other non-numeric values, e.g., strings, lists, etc.

        # If the value is NaN or not a valid number, return a default integer value (0, -1, etc.)
        return 0  # Replace with your desired default value
