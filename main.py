import torch
import copy
import torch.nn as nn
import matplotlib.pyplot as plt
from torch import optim
from Dataset import build_dataset
from model import mymodel

X_train, y_train, X_test, y_test, scaler = build_dataset(
    csv_path="./data/sine.csv",
    window_size=20,
    train_ratio=0.8
)


model = mymodel(input_size=1, hidden_size=10, num_layers=1, output_size=1)
optimizer = optim.Adam(model.parameters(), lr=0.001)
Loss_fn = nn.MSELoss()

best_final_loss = float("inf")
best_final_w = None
cnt = 0
Loss_all = 0


for epoch in range(200):
    model.train()
    Loss_all = 0

    for bx, by in zip(X_train, y_train):
        bx = bx.unsqueeze(0)
        by = by.unsqueeze(0)

        y_pred = model(bx)
        loss = Loss_fn(y_pred, by)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        Loss_all += loss.item()

    avg_loss = Loss_all / len(X_train)

    if epoch % 10 == 0:
        print("epoch:", epoch, "loss:", avg_loss)

    if epoch % 2 == 0:
        model.loss_history.append(avg_loss)
        model.epoch_record.append(epoch)

    if avg_loss < best_final_loss:
        best_final_loss = avg_loss
        best_final_w = copy.deepcopy(model.state_dict())
        cnt = 0
    else:
        cnt += 1
        if cnt >= 50:
            print(f"Final model early stop epoch {epoch}")
            break


model.load_state_dict(best_final_w)
model.eval()

with torch.no_grad():
    pred = model(X_test)
    loss_3 = Loss_fn(pred, y_test).item()
    print("test_loss:", loss_3)

    preds = pred.squeeze().numpy()
    truths = y_test.squeeze().numpy()


    preds_raw = scaler.inverse_transform(preds.reshape(-1, 1)).squeeze()
    truths_raw = scaler.inverse_transform(truths.reshape(-1, 1)).squeeze()


    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'SimSun']
    plt.rcParams['axes.unicode_minus'] = False

    plt.figure()
    plt.plot(truths_raw, label="真实序列", marker="o", markersize=3)
    plt.plot(preds_raw, label="模型预测", marker="x", markersize=3)
    plt.xlabel("时间")
    plt.ylabel("数值")
    plt.title("测试集：真实序列 vs 模型预测")
    plt.legend()
    plt.savefig("prediction.png", dpi=150)
    plt.show()

plt.figure()
plt.scatter(model.epoch_record, model.loss_history, s=20)
plt.xlabel("epoch")
plt.ylabel("loss")
plt.ylim(bottom=0, top=0.05)
plt.title("RNN Training Loss")
plt.grid(alpha=0.3)
plt.show()
