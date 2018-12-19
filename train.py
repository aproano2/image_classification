import json
import helper
import argparse


def main():

    ap = argparse.ArgumentParser(description='train.py')
    ap.add_argument('data_dir', nargs='*', action="store", default="./flowers")
    ap.add_argument('--save_dir', dest="save_dir", action="store", default="./checkpoint.pth")
    ap.add_argument('--number_epochs', dest="number_epochs", action="store", type=int, default=1)
    ap.add_argument('--model_type', dest="model_type", action="store", default="vgg16", type = str)
    ap.add_argument('--input_class', type=int, dest="input_class", default=760, action="store")
    

    train_dir = ap.data_dir + '/train'
    valid_dir = ap.data_dir + '/valid'
    test_dir = ap.data_dir + '/test'

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    batch_size = 64
    output_class = 102
    
    trainloader, traindata = helper.train_data_loader(train_dir, mean, std, batch_size)
    validloader = helper.test_data_loader(valid_dir, mean, std, batch_size)
    testloader = helper.test_data_loader(test_dir, mean, std, batch_size)

    model, criterion, optimizer = helper.build_network(ap.model_type, input_class=ap.input_class, output_class=output_class)
    model = helper.train_model(model, criterion, optimizer, number_epochs=ap.number_epochs, print_every=10)
    helper.save_model(ap.save_dir, model, train_data, ap.model_type, ap.input_class, output_class)
    

if __name__ = '__main__':
    main()
