import os
from collections import namedtuple

import cv2
import geopandas as gpd
import numpy as np
import pandas as pd
from PIL import Image

from shapely.geometry import Polygon, Point
#from qgis.core import QgsProject, QgsMapLayerRegistry, QgsPointXY, QgsGeometry, QgsFeature, QgsSymbol, QgsMarkerSymbolLayer

from LabelingTools import Labels



def extract_color(img, target_color):
    img_bw = Image.new(
        "L", img.size
    )  # Create a new black and white image of the same size

    for x in range(img.width):
        for y in range(img.height):
            pixel_color = img.getpixel((x, y))
            if (pixel_color == target_color).any():
                 img_bw.putpixel((x, y), 255)  # Set pixel to white for the target color
            else:
                img_bw.putpixel((x, y), 0) # Set pixel to black for non-target colors


    return img_bw


def find_contours(img_bw):
    img_array = np.array(img_bw)
    _, threshold_img = cv2.threshold(img_array, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(
        threshold_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )
    return contours


def create_shapefile(
    shapefile_path, named_contours, crs, extent, x_resolution, y_resolution, id_counter
) -> int:
    # gdf = gpd.GeoDataFrame(columns=['id', 'tag', 'geometry'], crs=crs)

    gdf_list = []
    # id_counter = 0
    for named_contour in named_contours:
        for i, contours in enumerate(named_contour.Contours):
            # Extract the contour points as tuples

            points = [
                (
                    (point[0][0] * x_resolution) + extent.left,  # + 0.00001867)
                    (point[0][1] * y_resolution) + extent.bottom,
                )
                for point in contours
            ]

            if len(points) >= 4:
                polygon = Polygon(points)
                polygon = polygon.buffer(0, cap_style='square')


                # Create a GeoDataFrame for each polygon with data
                data = {"id": [id_counter], "LC": [named_contour.Name]}
                id_counter += 1
                gdf_temp = gpd.GeoDataFrame(data, geometry=[polygon], crs=crs)
                # Append the temporary GeoDataFrame to the list
                gdf_list.append(gdf_temp)

                # data ={'id': i, 'tag': 'Tag_' + str(i)}

                # gdf = gdf.append({'id': data['id'], 'tag': data['tag'], 'geometry': polygon}, ignore_index=True)

            else:
                print(
                    "\033[91mThe extracted points are less then 4. Contour points are left out.\033[0m"
                )



    # gdf.to_file(shapefile_path)
    gdf = gpd.GeoDataFrame(pd.concat(gdf_list, ignore_index=True), crs=crs)

    scale_factor_x = (extent.right - extent.left) / (gdf.total_bounds[2] - gdf.total_bounds[0])
    scale_factor_y = (extent.top - extent.bottom) / (gdf.total_bounds[3] - gdf.total_bounds[1])

    # Calculate the center of the image
    center_x = extent.left
    center_y = extent.bottom

    scaled_geometries = gdf.scale(xfact=scale_factor_x, yfact=scale_factor_y, origin=(center_x, center_y))
    scaled_gdf = gpd.GeoDataFrame(geometry=scaled_geometries)

    # Copy the attribute data from the original GeoDataFrame to the scaled one
    scaled_gdf = scaled_gdf.join(gdf.drop('geometry', axis=1))

    scaled_gdf.to_file(f"{shapefile_path}\SHP.shp")
    return id_counter
    
def create_shapefile_for_POI(shapefile_path, named_contours, crs, extent, x_resolution, y_resolution, id_counter) -> int:
    gdf_list = []
    # id_counter = 0
    for named_contour in named_contours:  # All the contours correspondng to one name.
        for i, contours in enumerate(named_contour.Contours):  # one particular polygon's surrounding contour points
            # each contours is the surrounding of one point

            point_x = np.mean([point[0][0] for point in contours])  # find the middle point on the x and y axis
            point_y = np.mean([point[0][1] for point in contours])

            point_x = (point_x * x_resolution) + extent.left  # origin point
            point_y = (point_y * y_resolution) + extent.bottom  # TODO Get origin point straight from the source image


            point = Point(point_x, point_y)

            if named_contour.Name != -1:
                # Create a GeoDataFrame for each polygon with data
                data = {"id": [id_counter], "Objektum": [named_contour.Name]}
                id_counter += 1
                gdf_temp = gpd.GeoDataFrame(data, geometry=[point], crs=crs)
                # Append the temporary GeoDataFrame to the list
                gdf_list.append(gdf_temp)

            # data ={'id': i, 'tag': 'Tag_' + str(i)}

            # gdf = gdf.append({'id': data['id'], 'tag': data['tag'], 'geometry': polygon}, ignore_index=True)



    # gdf.to_file(shapefile_path)
    if len(gdf_list) > 0:
        for gdf in gdf_list:
            if gdf.geometry.type[0] != 'Point':
                # If geometry is not Point, convert it to Point
                gdf['geometry'] = gdf['geometry'].centroid

        gdf = gpd.GeoDataFrame(pd.concat(gdf_list, ignore_index=True), crs=crs)

        #scale_factor_x = (extent.right - extent.left) / (gdf.total_bounds[2] - gdf.total_bounds[0])
        #scale_factor_y = (extent.top - extent.bottom) / (gdf.total_bounds[3] - gdf.total_bounds[1])

        # Calculate the center of the image
        #center_x = extent.left  # origin point
        #center_y = extent.bottom   #TODO Get origin point straight from the source image

        #scaled_geometries = gdf.scale(xfact=scale_factor_x, yfact=scale_factor_y, origin=(center_x, center_y))
        #scaled_gdf = gpd.GeoDataFrame(geometry=scaled_geometries)

        # Copy the attribute data from the original GeoDataFrame to the scaled one
        #scaled_gdf = scaled_gdf.join(gdf.drop('geometry', axis=1))

        gdf.to_file(f"{shapefile_path}\SHP.shp")
    return id_counter

def save_as_single_shapefile(input_image: np.ndarray, target_colors: list, crs_value, extent_value, x_resolution, y_resolution,id_counter, results_folder: str = "Results", is_POI: bool = False):

    input_image = cv2.flip(input_image, 0)

    #img = Image.fromarray(input_image)


    all_named_contours = []

    NamedContour = namedtuple("NamedContour", ["Name", "Contours"])
    labels = Labels(())
    labels.ReadJSON("Data/labels.json")
    for color in target_colors:
        #img_bw = extract_color(img, color)
        #contours = find_contours(img_bw)

        mask = cv2.inRange(input_image, color, color)

        kernel = np.ones((3, 3), np.uint8)
        dilated_mask = cv2.dilate(mask, kernel, iterations=1)

        # Find contours in the mask
        contours, _ = cv2.findContours(dilated_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


        Name = 0
        for label in labels.list:
            if np.array_equal(label.RGBColor, color):
                Name = label.ID
                if Name == -1:
                    print("check please")

                break

        nc = NamedContour(Name, contours)

        all_named_contours.append(nc)



    shapefile_path = f"{os.path.abspath(os.getcwd())}/{results_folder}"
    os.makedirs(shapefile_path, exist_ok=True)
    new_id_index = 0
    if crs_value is not None and not is_POI:
        new_id_index = create_shapefile(
            shapefile_path,
            all_named_contours,
            crs_value,
            extent_value,
            x_resolution,
            y_resolution,
            id_counter,
        )
    elif crs_value is not None and is_POI:
        new_id_index = create_shapefile_for_POI(
            shapefile_path,
            all_named_contours,
            crs_value,
            extent_value,
            x_resolution,
            y_resolution,
            id_counter,
        )
    else:
        print(f"\033[91mNo CRS value was given. Did not created shapefile at path {results_folder}\033[0m")
    return new_id_index


def collect_shps_to_one(path_list: list, result_place: str, is_POI: bool = False):
    # crs = CRS.from_epsg(23700)


    gdf_original = gpd.read_file(f"{path_list[0]}/SHP.shp")
    # gdf_original = gdf_original.set_crs(crs)

    for i in range(1, len(path_list)):
        gdf_next = gpd.read_file(f"{path_list[i]}/SHP.shp")
        # gdf_next = gdf_next.set_crs(crs)

        gdf_original = gpd.pd.concat(
            [gdf_original, gdf_next]
        )  # sjoin(gdf_original, gdf_next) #gdf_original.merge(gdf_next, on='ID')
        if not is_POI:
            gdf_original["geometry"] = gdf_original["geometry"].buffer(0)

    if not is_POI:
        gdf_original['geometry'] = gdf_original['geometry'].simplify(tolerance=0.001, preserve_topology=True)

    try:
        gdf_original.to_file(f"{result_place}/output.shp")
        print("Done")
        return gdf_original
    except Exception as e:
        print(e)


def save_png_as_shp(
    input_image: np.ndarray, target_colors: list, results_folder: str = "Results"
):
    # png_file_path = "Resources/contour.png"
    # target_colors = [
    #    (237, 28, 36),
    #    (163, 73, 164),
    #    (255, 242, 0),
    #    (0, 162, 232),
    #    (0, 0, 0),
    # ]  # Replace this with the colors you're looking for

    img = Image.fromarray(input_image)  # Image.open(png_file_path)

    img_bw_path = f"{os.path.abspath(os.getcwd())}/{results_folder}/bw"
    os.makedirs(img_bw_path, exist_ok=True)

    shapefile_path = f"{os.path.abspath(os.getcwd())}/{results_folder}/SHP"
    os.makedirs(shapefile_path, exist_ok=True)

    for color in target_colors:
        img_bw = extract_color(img, color)
        img_bw.save(f"{img_bw_path}/{color}_bw.png")  # Create folders?

        contours = find_contours(img_bw)
        create_shapefile(
            f"{shapefile_path}/{color}_borders.shp", contours
        )  # Create folders?
