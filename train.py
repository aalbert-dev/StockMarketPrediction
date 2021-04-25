import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
import torch.utils.data as Data
import pandas as pd
from sklearn.model_selection import train_test_split

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

input_size = 5  # rnn input size
output_size = 1
lr = 0.02
batch_size = 512
num_epochs = 200
seq_length = 365
threshold = 0.02


class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()

        self.rnn = nn.LSTM(
            input_size=input_size,
            hidden_size=128,  # rnn hidden unit
            num_layers=3,  # number of rnn layer
            batch_first=True,  # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
        )
        self.out = nn.Linear(128, output_size)
        self.relu = nn.ReLU()

    def forward(self, x, h_state):
        # x (batch, time_step, input_size)
        # h_state (n_layers, batch, hidden_size)
        # r_out (batch, time_step, hidden_size)
        r_out, h_state = self.rnn(x, h_state)
        # else:
        outs = []  # save all predictions
        for time_step in range(r_out.size(1)):  # calculate output for each time step
            outs.append(self.out(r_out[:, time_step, :]))
        out = torch.stack(outs, dim=1)
        out = self.relu(out)

        return out, h_state


def train(symbol="VZ"):
    """
    This is a sample training code.

    Warning:
    Do not use it directly. Use as a reference to design/modify your own training codes to fit your own model.
    """
    df = pd.read_csv(f"./data/{symbol}.csv", parse_dates=["Date"]).sort_values(by="Date")

    df["Close"] = df["Close"] / 100
    df["prev1"] = df["Close"].shift(1)
    df["prev2"] = df["Close"].shift(2)
    df["prev3"] = df["Close"].shift(3)
    df["prev4"] = df["Close"].shift(4)
    df["prev5"] = df["Close"].shift(5)
    df = df.dropna(subset=["prev5"]).reset_index(drop=True)
    x = df[["prev1", "prev2", "prev3", "prev4", "prev5"]].to_numpy(dtype=np.float32)
    y = df["Close"].to_numpy(dtype=np.float32).reshape(-1, 1)
    print(str(x.shape) + " " + str(y.shape) + "\n")

    xs = []
    ys = []
    for i in range(0, len(x) - seq_length + 1, 1):
        xs.append(x[i : i + seq_length, :].copy())
        ys.append(y[i : i + seq_length, :].copy())
    xs = np.array(xs)
    ys = np.array(ys)
    print(str(xs.shape) + " " + str(ys.shape) + "\n")

    x_train, x_test, y_train, y_test = train_test_split(xs, ys, test_size=0.25)

    x_train = torch.from_numpy(x_train)
    y_train = torch.from_numpy(y_train)
    print("Train: " + str(x_train.shape) + " " + str(y_train.shape) + "\n")

    train_dataset = Data.TensorDataset(x_train, y_train)

    train_loader = Data.DataLoader(
        dataset=train_dataset,  # torch TensorDataset format
        batch_size=batch_size,  # mini batch size
        shuffle=True,  # random shuffle for training
    )

    x_test = torch.from_numpy(x_test)
    y_test = torch.from_numpy(y_test)
    print("Test: " + str(x_test.shape) + " " + str(y_test.shape) + "\n")

    test_dataset = Data.TensorDataset(x_test, y_test)
    test_loader = Data.DataLoader(
        dataset=test_dataset,  # torch TensorDataset format
        batch_size=batch_size,  # mini batch size
        shuffle=True,  # random shuffle for training
    )

    rnn = RNN().to(device)
    optimizer = torch.optim.Adam(rnn.parameters(), lr=lr)  # optimize all cnn parameters
    loss_func = nn.MSELoss()
    total_step = len(train_loader)

    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            images = (
                images.view(-1, seq_length, input_size).type(torch.float).to(device)
            )
            labels = (
                labels.view(-1, seq_length * output_size).type(torch.float).to(device)
            )
            # Forward pass
            outputs, _ = rnn(images, h_state=None)
            loss = loss_func(outputs.reshape(-1, seq_length * output_size), labels)
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()  # retain_graph=True
            # torch.nn.utils.clip_grad_norm_(rnn.parameters(), clip)
            optimizer.step()
            if i == 0:
                print(
                    "Epoch: "
                    + str(epoch)
                    + "| Step "
                    + str(i)
                    + " | Loss: "
                    + str(loss)
                )

            if epoch % 10 == 0 and i == 0:
                for j, (t_x, t_y) in enumerate(test_loader):
                    t_x = (
                        t_x.view(-1, seq_length, input_size)
                        .type(torch.float)
                        .to(device)
                    )
                    t_y = (
                        t_y.view(-1, seq_length * output_size)
                        .type(torch.float)
                        .to(device)
                    )

                    test_output, _ = rnn(t_x, h_state=None)
                    test_loss = loss_func(
                        test_output.reshape(-1, seq_length * output_size), t_y
                    )
                    t_out = (
                        test_output.cpu()
                        .data.numpy()
                        .reshape(-1, seq_length * output_size)
                    )
                    t_y = t_y.cpu().data.numpy().reshape(-1, seq_length * output_size)
                    correct = np.count_nonzero(
                        np.absolute(t_out - t_y) <= threshold
                    ) / (t_y.shape[0] * t_y.shape[1])
                    if j == 0:
                        print(
                            "Epoch: "
                            + str(epoch)
                            + "| Test Step "
                            + str(j)
                            + " | Val_Loss: "
                            + str(test_loss)
                            + " | CR: "
                            + str(correct)
                            + "\n"
                        )

        if epoch % 100 == 0:
            torch.save(rnn, "./models/model_" + str(epoch) + ".pkl")
            print("Model saved\n")

    torch.save(rnn, f"./models/model_{num_epochs}.pkl")

if __name__ == "__main__":
    train()
