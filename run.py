import sys

def main(args):  
  
    from fewshot.models.aves_plus_labelencoder.train import main as train_model
    train_model(args)

if __name__ == "__main__":
    main(sys.argv[1:])