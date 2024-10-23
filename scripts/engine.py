# training loop
import torch as T
import torch.nn as nn
from torchmetrics import Accuracy
from tqdm.auto import tqdm
from torch.utils.tensorboard import SummaryWriter

device = "cuda" if T.cuda.is_available() else "cpu"
acc_fn = Accuracy(task="multiclass", num_classes=1000).to(device)


def train_step(model:nn.Module,
               train_dataloader:T.utils.data.DataLoader,
               loss_fn:nn.Module,
               optimizer:T.optim.Optimizer,
               device:T.device)->tuple[float,float]:
    
    ### TRAIN ###
    model.train()
    loss_value, acc_value = 0, 0
    for batch, (X, y) in enumerate(train_dataloader):
        # Predict y from X 
        X, y=X.to(device), y.to(device)
        pred_logit = model(X)
        loss = loss_fn(pred_logit,y)
        # Calculate the loss and acc
        loss_value+=loss.item()
        acc = acc_fn(T.softmax(pred_logit, dim=1).argmax(dim=1), y).item()
        acc_value += acc
        #
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #
    loss_value /= len(train_dataloader)
    acc_value /= len(train_dataloader)
    return loss_value, acc_value

# testing loop
def test_step(model:nn.Module,
              test_dataloader:T.utils.data.DataLoader,
              loss_fn:nn.Module,
              device:T.device) ->tuple[float,float] :
    ### Test ###
    model.eval()
    loss_val = 0
    acc_val = 0
    for batch, (X, y) in enumerate(test_dataloader):
        X, y = X.to(device), y.to(device)
        # make test prediction model
        test_pred_logit = model(X)
        #  calculate the loss and acc
        loss = loss_fn(test_pred_logit,y)
        loss_val += loss.item()
        acc = acc_fn(T.softmax(test_pred_logit, dim=1).argmax(dim=1), y).item()
        acc_val += acc
    loss_val /= len(test_dataloader)
    acc_val /= len(test_dataloader)
    return loss_val, acc_val

def train(model:nn.Module,
          train_dataloader:T.utils.data.DataLoader,
          test_dataloader:T.utils.data.DataLoader,
          loss_fn:nn.Module,
          optimizer:T.optim.Optimizer,
          epochs:int,
          device:T.device,
          writer:T.utils.tensorboard.SummaryWriter):
    for epoch in tqdm(range(epochs)):
        # Create empty results dictionary
        results = {"train_loss": [],
                "train_acc": [],
                "test_loss": [],
                "test_acc": []
        }
        model.to(device)
        train_loss, train_acc = train_step(model=model,
                                        train_dataloader=train_dataloader,
                                        loss_fn=loss_fn,
                                        optimizer=optimizer,
                                        device=device)
        test_loss, test_acc = test_step(model=model,
                                        test_dataloader=test_dataloader,
                                        loss_fn=loss_fn,
                                        device=device)
        ### EXPERIMENT ###
        writer.add_scalars(main_tag="Accuracy",
                          tag_scalar_dict={"train_acc":train_acc,
                                           "test_acc" :test_acc},
                            global_step = epoch)
        writer.add_scalars(main_tag="Loss",
                          tag_scalar_dict={"train_loss": train_loss,
                                           "test_loss": test_loss},
                                           global_step=epoch)
        writer.add_graph(model=model,
                         input_to_model=T.rand(32,3,224,224).to(device))
        writer.close()

        ### END ###
        print(
          f"Epoch: {epoch+1} | "
          f"train_loss: {train_loss:.4f} | "
          f"train_acc: {train_acc:.4f} | "
          f"test_loss: {test_loss:.4f} | "
          f"test_acc: {test_acc:.4f}"
        )
        # Update results dictionary
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)

    return results