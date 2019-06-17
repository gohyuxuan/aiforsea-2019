from data import Data
from model import Model
import pandas as pd

DATA_PATH = 'test_data.csv'

if __name__ == '__main__':
    data = pd.read_csv(DATA_PATH)
    data = Data(data)
    processed_data = data.get_processed_data()
    booking_ids = data.get_id_sequence()
    predictions = Model().predict(processed_data)
    result = {booking_id:prediction for booking_id, prediction in zip(booking_ids, predictions)}
    print(result)
