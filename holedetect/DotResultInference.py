import os
import rasterio
import numpy as np
from geopandas import overlay
from rasterio.crs import CRS
import geopandas as gpd
from shapely import Polygon

def euclidean_distance(point1, point2):
    point1 = np.array(point1)
    point2 = np.array(point2)
    return np.sqrt(np.sum((point1 - point2)**2))

def CreateInference(originalImages: str, originalSHP: str, createdSHP):
    images = os.listdir(originalImages)  # List of all image names in this subdirectory
    oriImg = list()
    for i, image_name in enumerate(images):
        if image_name.endswith(".tif"):  # Only read jpg images...
            oriImg.append(f"{originalImages}/{image_name}")

    left_points = list()
    right_points = list()
    top_points = list()
    bot_points = list()
    for img in oriImg:
        with rasterio.open(img, 'r+') as src:
            if src.crs is None:
                src.crs = CRS.from_epsg(23700)

            # Get the extent (bounding box) of the GeoTIFF
            bounds = src.bounds
            left_points.append(bounds.left)
            right_points.append(bounds.right)
            top_points.append(bounds.top)
            bot_points.append(bounds.bottom)


    # Select height
    height = 0
    left = 0
    right = 0
    for left_point in left_points:
        for right_point in right_points:
            height_candidate = np.abs(left_point - right_point)
            if height_candidate > height:
                height = height_candidate
                left = left_point
                right = right_point


    width = 0
    top = 0
    bottom = 0
    for top_point in top_points:
        for bot_point in bot_points:
            width_candidate = np.abs(top_point - bot_point)
            if width_candidate > width:
                width = width_candidate
                top = top_point
                bottom = bot_point

#save extent
#left, bottom, right, top = 0,1,2,3
    # cut the original shp with this extent
    cut_polygon = Polygon([
        # (left, bottom),
        # (left, top),
        # (right, top),
        # (right, bottom)
        (right, top),
        (right, bottom),
        (left, bottom),
        (left, top)
    ])

    gdf = gpd.read_file(originalSHP)
    gdf.crs = createdSHP.crs

    gdf['New_Id'] = range(1, len(gdf) + 1)
    gdf['id'] = gdf['New_Id']
    gdf = gdf.drop(columns=['New_Id'])

    gdf_cropped = overlay(gdf, gpd.GeoDataFrame({'geometry': [cut_polygon]}, crs=src.crs),
                          how='intersection')

    os.makedirs("inference", exist_ok=True)
    gdf_cropped.to_file("inference/fitted.shp")

    # compare original and your shp
    # compares = for each predicted point search for a point which's distance is under a criteria.
    original_points = [(geometry.x, geometry.y) for geometry in gdf_cropped['geometry']]
    my_points = [(geometry.x, geometry.y) for geometry in createdSHP['geometry']]

    closest_idxs = list()

    not_found_from_my_points = 0
    for pred in my_points:
        distances = [euclidean_distance(pred, ori) for ori in original_points]

        closest_index = np.argmin(distances)

        # Get the closest point from list2
        if np.min(distances) < 1.01:
            #closest_point = original_points[closest_index]
            # store which points are found store id of original and predicted.
            closest_idxs.append(gdf_cropped['id'][closest_index])
        else:
            not_found_from_my_points += 1

    # calculate accuracy for this.

    # TN - weird doesn't get calculated
    # TP - at what percentage do I find close points from the original data to my points
    TP = len(closest_idxs) / len(my_points)
    result_txt = "The percentage of True positives is: " + str(TP) + "\r\n"

    #FT - how many doesn't have a close enough corresponding element among my points
    FP = not_found_from_my_points / len(my_points)
    result_txt += "The percentage of false positives is: " + str(FP) + "\r\n"

    # FN - only pointed out on the original 1-(found / original)
    FN = 1 - (len(closest_idxs) / len(original_points))
    result_txt += "The percentage of false negative is: " + str(FN) + "\r\n"

    result_txt += "My points " + str(len(my_points)) + "\r\n"
    result_txt += "all points " + str(len(original_points)) + "\r\n"
    result_txt += "close points " + str(len(closest_idxs)) + "\r\n"
    result_txt += "not found among mine " + str(not_found_from_my_points) + "\r\n"
    print(result_txt)

    # Filter out the found points from the SHP and return that too (so you can check what you don't found)
    gdf_filtered = gdf_cropped[~gdf_cropped['id'].isin(closest_idxs)]
    gdf_filtered.to_file("inference/filtered.shp")



    return TP, FP, FN, result_txt








