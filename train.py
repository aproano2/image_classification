import json
import helper
import argparse


def main():

    ap = argparse.ArgumentParser(description='train.py')
    ap.add_argument('data_dir', nargs='*', action="store", default="./flowers")
    ap.add_argument('--save_dir', dest="save_dir", action="store", default="./checkpoint.pth")
    ap.add_argument('--number_epochs', dest="number_epochs", action="store", type=int, default=1)
    ap.add_argument('--model_type', dest="model_type", action="store", default="vgg16", type = str)
    ap.add_argument('--hidden_units', type=int, dest="hidden_units", default=760, action="store")
    ap.add_argument('--learning_rate', dest="learning_rate", action="store", default=0.001, type=float)
    ap.add_argument('--dropout', dest = "dropout", action = "store", default = 0.5)
    ap.add_argument('--gpu', dest="gpu", action='store_true', default = False)

    args = ap.parse_args()

    train_dir = args.data_dir + '/train'
    valid_dir = args.data_dir + '/valid'
    test_dir = args.data_dir + '/test'

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    batch_size = 64
    output_class = 102
    
    trainloader, traindata = helper.train_data_loader(train_dir, mean, std, batch_size)
    validloader = helper.test_data_loader(valid_dir, mean, std, batch_size)
    testloader = helper.test_data_loader(test_dir, mean, std, batch_size)

    model, criterion, optimizer = helper.build_network(args.model_type, hidden_units=args.hidden_units,
                                                       output_class=output_class, dropout=args.dropout, lr=args.learning_rate)
    model = helper.train_model(model, criterion, optimizer, trainloader, validloader, number_epochs=args.number_epochs, print_every=10, gpu=args.gpu)
    helper.save_model(args.save_dir, model, traindata, args.model_type, args.hidden_units, output_class, args.dropout, args.learning_rate)
    

if __name__ == '__main__':
    main()
