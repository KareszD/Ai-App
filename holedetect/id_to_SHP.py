import rasterio
from shapely.geometry import Point
import geopandas as gpd
import pandas as pd
from itertools import combinations
import numpy as np

class bboxProcessor:
    def __init__(self, images_boxes):
        self.images_boxes = images_boxes
        self.crs = None
        self.extent = None
        self.x_resolution = None
        self.y_resolution = None
        self.id_counter = 0
        self.all_gdf = []

    def process_results(self):
        path_to_shp = []
        for result in self.images_boxes:
            if len(result[1]) > 0:
                self._gather_geo_info(result[0])

                self._process_boxes(result[1])

        if len(self.all_gdf) > 0:
           return self._merge_gdf_into_one()


    def _gather_geo_info(self, image):
        with rasterio.open(image) as src:
            # Get the spatial reference information
            self.crs = src.crs
            self.extent = src.bounds
            self.x_resolution = src.res[0]
            self.y_resolution = src.res[1]

    def _process_boxes(self, boxes):
        gdf_list = []
        for box in boxes:
            # tuple (x, y)
            point_x = (box[0] * self.x_resolution) + self.extent.left #['left']
            point_y = (box[1] * self.y_resolution) + self.extent.bottom #['bottom']

            point = Point(point_x, point_y)

            #create shp
            data = {"id": [self.id_counter], "Objektum": ["Földi kutya üreg"]}
            self.id_counter += 1
            gdf_temp = gpd.GeoDataFrame(data, geometry=[point], crs=self.crs)

            gdf_list.append(gdf_temp)

        if len(gdf_list) > 0:
            for gdf in gdf_list:
                if gdf.geometry.type[0] != 'Point':
                    # If geometry is not Point, convert it to Point
                    gdf['geometry'] = gdf['geometry'].centroid

            gdf = gpd.GeoDataFrame(pd.concat(gdf_list, ignore_index=True), crs=self.crs)
            self.all_gdf.append(gdf)

    def _merge_gdf_into_one(self):
        gdf_original = self.all_gdf[0]

        for i in range(1, len(self.all_gdf)):
            gdf_original = gpd.pd.concat([gdf_original, self.all_gdf[i]])

        # threshold = 0.4
        #
        # indices_to_remove = []
        #
        # for i, j in combinations(gdf_original.index, 2):
        #     point1 = gdf_original.loc[i, 'geometry']
        #     point2 = gdf_original.loc[j, 'geometry']
        #
        #     distance = point1.distance(point2)
        #
        #     if np.all(distance < threshold):
        #         indices_to_remove.append(i)
        #         break
        #
        # gdf_original = gdf_original.drop(indices_to_remove)

        try:
            result_shp_path = f"out/output.shp"
            gdf_original.to_file(result_shp_path)
            return gdf_original
            print("Done!")
        except Exception as e:
            print(e)

        return gpd.GeoDataFrame()

