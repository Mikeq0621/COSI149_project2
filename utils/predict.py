import torch
from utils.model import RNN
from utils.model import output_size
import numpy


class Predictor:
    def __init__(self, seq_length, input_size, model_file):
        self.seq_length = seq_length
        self.input_size = input_size
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model = RNN()
        self.model = torch.load(model_file, map_location=self.device)
        self.model.eval()

    def predict(self, data_loader, scaler=None):
        for j, (x, y) in enumerate(data_loader):
            t_x = x
            t_y = y

        t_x = t_x[0]

        last_day_close = t_x[-1][0:5].reshape(-1, 5)
        print("predict_x: " + str(scaler.inverse_transform(last_day_close)))

        t_x = t_x.view(-1, self.seq_length, self.input_size).type(torch.float).to(self.device)
        test_output, _ = self.model(t_x, h_state=None)
        t_out = test_output.cpu().data.numpy().reshape(-1, output_size)
        if scaler is not None:
            t_out = scaler.inverse_transform(t_out)
        y_predict = [t_out[-1]]
        return y_predict
