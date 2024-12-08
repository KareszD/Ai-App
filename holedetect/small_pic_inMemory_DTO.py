class small_tif_data_container:
    def __init__(self, data, crs, bounds, res_x, res_y, x,y ):
        self.data = data
        self.crs = crs
        self.bounds = bounds
        self.x_resolution = res_x
        self.y_resolution = res_y
        self.x = x
        self.y = y