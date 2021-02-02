import jittor as jt 
import jittor.nn as nn 
from dataset import TsinghuaDog
from jittor import transform
from jittor.optim import Adam, SGD
from tqdm import tqdm
import numpy as np
from model import Net
import argparse 


jt.flags.use_cuda=1

def train(model, train_loader, optimizer, epoch):
    model.train() 
    total_acc = 0
    total_num = 0
    losses = 0.0
    pbar = tqdm(train_loader, desc=f'Epoch {epoch} [TRAIN]')
    for images, labels in pbar:
        output = model(images)
        loss = nn.cross_entropy_loss(output, labels)
        optimizer.step(loss) 
        pred = np.argmax(output.data, axis=1)
        acc = np.mean(pred == labels.data) * 100 
        total_acc += acc
        total_num += labels.shape[0] 
        losses += loss
        pbar.set_description(f'Epoch {epoch} [TRAIN] loss = {loss.data[0]:.2f}, acc = {acc:.2f}')


best_acc = -1.0
def evaluate(model, val_loader, epoch=0, save_path='./best_model.pkl'):
    model.eval()
    global best_acc
    total_acc = 0
    total_num = 0

    for images, labels in val_loader:
        output = model(images)
        pred = np.argmax(output.data, axis=1)
        acc = np.sum(pred == labels.data)
        total_acc += acc
        total_num += labels.shape[0] 
    acc = total_acc / total_num 
    if acc > best_acc:
        best_acc = acc
        model.save(save_path)
    print ('Test in epoch', epoch, 'Accuracy is', acc, 'Best accuracy is', best_acc)

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--num_classes', type=int, default=130)

    parser.add_argument('--lr', type=float, default=2e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-5)

    parser.add_argument('--resume', type=bool, default=False)
    parser.add_argument('--eval', type=bool, default=False)

    parser.add_argument('--dataroot', type=str, default='/home/gmh/dataset/TsinghuaDog/')
    parser.add_argument('--model_path', type=str, default='./best_model.pkl')

    args = parser.parse_args()
    
    transform_train = transform.Compose([
        transform.Resize((512, 512)),
        transform.RandomCrop(448),
        transform.RandomHorizontalFlip(),
        transform.ToTensor(),
        transform.ImageNormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    root_dir = args.dataroot
    train_loader = TsinghuaDog(root_dir, batch_size=16, train=True, part='train', shuffle=True, transform=transform_train)
    
    transform_test = transform.Compose([
        transform.Resize((512, 512)),
        transform.CenterCrop(448),
        transform.ToTensor(),
        transform.ImageNormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    val_loader = TsinghuaDog(root_dir, batch_size=16, train=False, part='val', shuffle=False, transform=transform_test)
    epochs = args.epochs
    model = Net(num_classes=args.num_classes)
    lr = args.lr
    weight_decay = args.weight_decay
    optimizer = SGD(model.parameters(), lr=lr, momentum=0.9) 
    if args.resume:
        model.load(args.model_path)
    if args.eval:
        evaluate(model, val_loader)
        return 
    for epoch in range(epochs):
        train(model, train_loader, optimizer, epoch)
        evaluate(model, val_loader, epoch)


if __name__ == '__main__':
    main()

