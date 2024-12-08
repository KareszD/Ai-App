import json
from json import JSONEncoder

import numpy as np


class Label:
    def __init__(self, Name: str, Color, Id) -> None:
        self.__name = Name
        self.__id = Id
        if isinstance(Color, tuple):
            self.__color = np.array(Color)
        elif isinstance(Color, str):
            colorstr = Color.lstrip("#")
            self.__color = np.array(
                tuple(int(colorstr[i : i + 2], 16) for i in (0, 2, 4))
            )  # 110, 193, 228
        else:
            raise ValueError("Color is not in a correct format")

    @property
    def Name(self):
        return self.__name

    @Name.setter
    def Name(self, val):
        self.__name = val

    @property
    def ID(self):
        return self.__id

    @ID.setter
    def ID(self, val):
        self.__id = val

    @property
    def RGBColor(self):
        return self.__color

    @RGBColor.setter
    def RGBColor(self, val):
        if isinstance(val, tuple):
            self.__color = np.array(val)
        else:
            raise ValueError("Color is not in a correct format")

    @property
    def HexaColor(self):
        return "#" + ("%02x%02x%02x" % tuple(self.__color))

    @HexaColor.setter
    def HexaColor(self, val):
        if isinstance(val, str):
            colorstr = val.lstrip("#")
            self.__color = np.array(
                tuple(int(colorstr[i : i + 2], 16) for i in (0, 2, 4))
            )  # 110, 193, 228
        else:
            raise ValueError("Color is not in a correct format")

    def __eq__(self, other):
        return self.Name == other.Name

    def __lt__(self, other):
        return self.Name < other.Name

    # Define the greater than comparison method
    def __gt__(self, other):
        return self.Name > other.Name

    # Define the less than or equal to comparison method
    def __le__(self, other):
        return self.Name <= other.Name

    # Define the greater than or equal to comparison method
    def __ge__(self, other):
        return self.Name >= other.Name


    # Define the inequality comparison method
    def __ne__(self, other):
        return self.Name != other.Name

    def __hash__(self):
        # We use the hash value of the id attribute to make the object hashable
        return hash(self.ID)


class Labels:
    def __init__(self, Labels: list) -> None:
        self.__list = Labels

    def ToJSON(self, path: str):
        json_data = [
            {"name": label.Name, "color": label.HexaColor, "Id": label.ID} for label in self.__list
        ]
        with open(path, "w") as write:
            json.dump(json_data, write)  # cls=LabelEncoder

    def ReadJSON(self, path: str):
        with open(path, "r") as file:
            deserialized_labels = json.load(file)
        self.__list = [
            Label(label["name"], label["color"], label["Id"]) for label in deserialized_labels
        ]

    @property
    def list(self):
        return self.__list

    def add(self, label: Label):
        self.__list.append(label)

    def remove(self, label: Label):
        self.__list.remove(label)


# class LabelEncoder(JSONEncoder):
#    def default(self, o):
#        return o.__dict__
