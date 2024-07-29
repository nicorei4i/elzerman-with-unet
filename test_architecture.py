#%%
import torch
import h5py
import matplotlib.pyplot as plt
from model import UNet

Y_TRAIN_PATH = "sim_elzerman_traces_train.hdf5"


with h5py.File(Y_TRAIN_PATH, "r") as f:
    #print(f.keys())
    Y_data = torch.tensor(f["trace_0"][:]).long()
    
Y_data = (Y_data + 1) // 2
print(Y_data.shape)


X_data = Y_data.clone().float()
X_data = X_data + torch.randn_like(X_data) * 0.5

plt.plot(X_data)
plt.plot(Y_data)
plt.show()

model = UNet()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
loss_fn = torch.nn.CrossEntropyLoss()

X = X_data.unsqueeze(0).float()
Y = Y_data.unsqueeze(0).long()

print(X.shape)
print(Y.shape)


losses = []
for i in range(1000):
    print(f"Epoch {i}, Loss {losses[-1] if len(losses) > 0 else None}", end="\r")
    optimizer.zero_grad()
    Y_pred = model(X).unsqueeze(0)
    loss = loss_fn(Y_pred, Y)
    loss.backward()
    optimizer.step()
    
    losses.append(loss.item())

plt.plot(losses)
plt.show()

m = torch.nn.Softmax(dim=1)
Y_pred = m(Y_pred)
    
Y_pred_npy = Y_pred.detach().numpy()[0]

# plt.plot(Y_pred_npy[0])
# plt.plot(Y_pred_npy[1])

print(Y_pred_npy.shape)

predition_class = Y_pred_npy.argmax(axis=0)

plt.plot(X_data)
plt.plot(Y_data)
plt.plot(predition_class)
plt.show()

plt.plot(Y_pred_npy[0], label="Class 0 Logits")
plt.plot(Y_pred_npy[1], label="Class 1 Logits")
plt.legend()
plt.show()
