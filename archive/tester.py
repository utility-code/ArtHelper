import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

current_best_score = 0

def test(model, device, test_loader):
    global current_best_score
    model.eval() # Setting model to test
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in tqdm(test_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += nn.CrossEntropyLoss()(output, target).item()
            pred = output.argmax(dim = 1 , keepdim= True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    acc =100. * correct / len(test_loader.dataset)
    state = {
        'net': model.state_dict(),
        'acc': acc,
    }
    if acc >= current_best_score:
        print(acc, current_best_score)
        torch.save(state, './models/model.pt')
        print("saved model")
        current_best_score = acc
    else:
        print("not better yet")

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{}({:.0f}%)\n'.format(test_loss, correct, len(test_loader.dataset),100. * correct / len(test_loader.dataset)))
    if acc >= 90:
        exit()
