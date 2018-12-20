import json
import helper
import argparse


def main():

    ap = argparse.ArgumentParser(description='predict.py')
    ap.add_argument('image_path', nargs='*', action="store", default="./flowers", type=str)
    ap.add_argument('checkpoint', action="store", type=str)
    ap.add_argument('--top_k', dest="top_k", action="store", type=int, default=1)
    ap.add_argument('--category_names', dest="category_names", action="store", type = str)
    ap.add_argument('--gpu', dest="gpu", action='store_true')

    args = ap.parse_args()
    
    if args.category_names:
        with open(args.category_names, 'r') as f:
            cat_to_name = json.load(f)
    else:
        cat_to_name = None
    model = helper.load_model(args.checkpoint)
    ps, cl = helper.predict(args.image_path[0], model, args.top_k, args.gpu)
    helper.show_results(ps, cl, args.top_k, cat_to_name)
    
if __name__ == '__main__':
    main()