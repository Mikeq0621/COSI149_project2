import utils.loader as loader
from utils.model import RNN, input_size, seq_length
from utils.predict import Predictor
import os
import pandas as pd

if __name__ == "__main__":
    data_dir = "../data"
    stock_list = ['VZ', 'T', 'WMT', 'MGM',  'GT', 'BBY', 'AFG', 'ERJ', 'MYE', 'ECPG', 'GCO', 'MPC', 'TRI', 'UFI']
    stock_no_model = ['GPS']
    # stock_lst = ['VZ']
    for stock in stock_list:
        print()
        print("------------------------------------------------------------")
        print("predict: " + stock)

        stock_file = os.path.join(data_dir, stock + ".csv")
        nas_file_path = "../data/IXIC.csv"

        model_file_dir = os.path.join("../models", stock)
        model_file = os.path.join(model_file_dir, os.listdir(model_file_dir)[0])

        data_loader, scalar = loader.from_file_to_data_loader(stock_file, nas_file_path)

        predictor = Predictor(seq_length=seq_length, input_size=input_size, model_file=model_file)
        predict = predictor.predict(data_loader, scalar)

        last_day_close_price = loader.read_last_item(stock_file)

        result = [last_day_close_price[0]] + [x for x in predict[0]]
        for i in range(len(result)):
            result[i] = ["T + " + str(i), result[i]]
        result_df = pd.DataFrame(data = result, columns=["Date", "Close"])
        print(result_df)

        result_df.to_csv("../output/predict_" + stock + ".csv", index=False)



        print("predict result: " + stock_file)
        print(result)
        print("------------------------------------------------------------")
        print()
