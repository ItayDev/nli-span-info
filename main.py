import argparse
import os
from torch import nn
from functools import partial
from multiprocessing import freeze_support
from transformers import RobertaConfig, RobertaModel, RobertaTokenizer
from training import train, configure_adam_optimizer
from models import LangInferModel
from data.data_loader import TrainLoaderFactory, MODE
from data.prepare_data import prepare_data
from data.collate import collate


def build_parser():
    parser = argparse.ArgumentParser(description="Training")
    parser.add_argument("--epoch_num", type=int, default=100, help="number of epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument("--lr", type=float, default=2e-5, help="learning rate")
    parser.add_argument("--workers", type=int, default=0, help="num workers for dataloader")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-9, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--snapshots_path", default='snapshots', type=str, help="path to save checkpoints")
    parser.add_argument("--span_drop", default=0.85, type=float, help="span_drop hyper-parameter value")
    parser.add_argument("--max_spans", default=1500, type=int, help="max amount of spans allowed to be generated")
    parser.add_argument("--span_heads", default=6, type=int, help="The number of span heads")
    parser.add_argument("--dataset", default='all', type=str, help="what datasets to train on. Can be one of: snli, "
                                                                   "mnli, all")

    return parser


def build_data_loaders(args, tokenizer):
    snli, mnli = prepare_data(tokenizer)
    injected_collate = partial(collate,
                               span_drop=args.span_drop,
                               max_spans=args.max_spans,
                               padding_token_id=tokenizer.pad_token_id)

    train_loader_factory = TrainLoaderFactory(args.batch_size, snli, mnli, injected_collate)

    train_all = train_loader_factory.get_loader(MODE.TRAIN_ALL)
    train_snli = train_loader_factory.get_loader(MODE.TRAIN_SNLI)
    train_mnli = train_loader_factory.get_loader(MODE.TRAIN_MNLI)
    test_snli = train_loader_factory.get_loader(MODE.TEST_SNLI)
    test_mnli_matched = train_loader_factory.get_loader(MODE.TEST_MNLI_MATCHED)
    test_mnli_mismatched = train_loader_factory.get_loader(MODE.TEST_MNLI_MISMATCHED)

    if args.dataset == 'all':
        return train_all, {
            'snli': test_snli,
            "mnli_matched": test_mnli_matched,
            "mnli_mismatched": test_mnli_mismatched
        }
    elif args.dataset == 'snli':
        return train_snli, {
            'snli': test_snli
        }
    elif args.dataset == 'mnli':
        return train_mnli, {
            "mnli_matched": test_mnli_matched,
            "mnli_mismatched": test_mnli_mismatched
        }
    else:
        raise ValueError("--dataset has to contain one of: all, mnli or snli")

def main():
    args = build_parser().parse_args()

    print("Creating snapshot directory if not exist...")
    if not os.path.exists(args.snapshots_path):
        os.mkdir(args.snapshots_path)

    print("Loading Roberta components...")
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    config = RobertaConfig.from_pretrained("roberta-base", output_hidden_states=True)
    base_model = RobertaModel(config).cuda()

    model = LangInferModel(base_model, config, args.span_heads).cuda()
    optimizer = configure_adam_optimizer(model, args.lr, args.weight_decay, args.adam_epsilon)
    print("Preparing the data for training...")
    train_loader, test_loaders = build_data_loaders(args, tokenizer)
    criterion = nn.CrossEntropyLoss()

    print(f"Training started for {args.epoch_num} epochs. Might take a while...")
    train(args.epoch_num, model, optimizer, criterion, train_loader, test_loaders, args.snapshots_path)
    print("Training is now finished. You can check out the results now")


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    freeze_support()
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
