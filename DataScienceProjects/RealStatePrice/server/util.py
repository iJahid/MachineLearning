import warnings
import json
import pickle
import numpy as np
__location = None
__data_columns = None
__model = None


def get_estimated_price(location, sqft, bath, bed):
    try:
        loc_column_index = __data_columns.index(location.lower())
    except:
        loc_column_index = -1

    x = np.zeros(len(__data_columns))
    x[0] = bath
    x[1] = bed
    x[2] = sqft
    if loc_column_index >= 0:
        x[loc_column_index] = 1

    return round(__model.predict([x])[0], 2)


def get_location_names():
    return __location


def load_saved_artiefacts():
    print('loading... artifacts')
    global __location
    global __data_columns
    global __model
    with open("./artifacts/columns.json", 'r') as f:
        __data_columns = json.load(f)['data_columns']
        __location = __data_columns[3:]

    with open("./artifacts/appartment_priceing.pickle", 'rb') as f:
        __model = pickle.load(f)

    print('loaded... artifacts Done')


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    load_saved_artiefacts()
    # print('ramagondanahalli', get_estimated_price(
    #     'ramagondanahalli', 2100, 2, 3))
    # print('ramamurthy nagar', get_estimated_price(
    #     'ramamurthy nagar', 2100, 2, 3))
    # print('rayasandra', get_estimated_price('rayasandra', 2100, 2, 3))
    # print('sahakara nagar', get_estimated_price('sahakara nagar', 2100, 2, 3))

    # print(get_location_names())
