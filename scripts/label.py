import pandas
import os


def label_data(stock_symbol, data_file_path, day_sequence_len=5, label='Close'):
    my_data = pandas.read_csv(data_file_path)


    i = 0

    close_price = float(my_data.iloc[i][label])
    record = [stock_symbol, 0, 0, 0, 0, 0]

    for j in range(i + 1, i + day_sequence_len + 1):
        if j >= len(my_data):
            break
        next_close_price = float(my_data.loc[j, label])
        ratio = (next_close_price - close_price) / close_price
        if (ratio <= -0.05):
            record[1] = 1
        if ratio <= -0.02:
            record[2] = 1
        if -0.02 < ratio < 0.02:
            record[3] = 1
        if ratio >= 0.02:
            record[4] = 1
        if ratio >= 0.05:
            record[5] = 1
    return record


if __name__ == "__main__":
    data_source = '../output/'
    output_path = '../output/'

    stock_list = ['VZ', 'T', 'WMT', 'MGM', 'GT', 'BBY', 'AFG', 'ERJ', 'MYE', 'ECPG', 'GCO', 'MPC', 'TRI', 'UFI']

    column = ['Stock_Symbol', '-2', '-1', '0', '1', '2']
    result = pandas.DataFrame(columns=column)


    count = 0
    for stock_name in stock_list:
        print("process: " + stock_name)
        labeled_data = label_data(stock_name, data_source + "predict_" + stock_name + ".csv")
        df = pandas.DataFrame([labeled_data], columns=column)
        result = result.append(df, ignore_index=True)
        count += 1

    result.to_csv(output_path + "label_data" + ".csv", index=False)
