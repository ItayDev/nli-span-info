import torch
import time
import matplotlib.pyplot as plt
from numpy import linspace


def configure_adam_optimizer(model, lr, weight_decay, adam_epsilon):
    no_decay = ["bias", "LayerNorm.weight"]
    roberta_parameters = model.transformer.named_parameters()
    model_parameters = [
        model.span_info_collect.parameters(),
        model.span_info_extract.parameters(),
        model.output.parameters()
    ]

    # This allows to put a minimal lr on Roberta to try to avoid over-fitting
    optimizer_grouped_parameters = [
                                       {
                                           'params': [p for n, p in roberta_parameters if
                                                      not any(nd in n for nd in no_decay)],
                                           'weight_decay': weight_decay,
                                           'lr': 0.0001
                                       },
                                       {
                                           'params': [p for n, p in model.transformer.named_parameters() if
                                                      any(nd in n for nd in no_decay)],
                                           'weight_decay': 0.0,
                                           'lr': 0.0001
                                       },
                                   ] + list(map(lambda parameters: {
        'params': list(parameters),
        'lr': lr
    }, model_parameters))

    return torch.optim.Adam(optimizer_grouped_parameters,
                            betas=(0.9, 0.98),  # according to RoBERTa paper
                            lr=0.05,
                            eps=adam_epsilon)


def train(epoch_num,
          model,
          optimizer,
          loss_function,
          train_data_loader,
          test_data_loaders,
          snapshot_path):
    start_time = time.time()
    train_losses = []
    evaluation_metrics_list = {}

    for dataset in test_data_loaders:
        evaluation_metrics_list[dataset] = {'accuracies': [], 'losses': []}

    for epoch in range(epoch_num):
        running_loss = 0.0
        for i, data in enumerate(train_data_loader):
            input_ids, start_indexes, end_indexes, labels = data
            input_ids = input_ids.cuda()
            start_indexes = start_indexes.cuda()
            end_indexes = end_indexes.cuda()
            labels = labels.cuda().squeeze(-1)
            optimizer.zero_grad()
            y_hats = model(input_ids, start_indexes, end_indexes)
            running_loss += loss_function(y_hats, labels.view(-1))
            optimizer.step()

            if i % 1000 == 0:
                running_loss /= 1000
                train_losses.append(running_loss)
                running_loss = 0

        metrics = {}

        for data_set in test_data_loaders:
            accuracy, loss = eval_model(model, test_data_loaders[data_set], loss_function)
            metrics[data_set] = {'loss': loss, 'accuracy': accuracy}
            evaluation_metrics_list['losses'].append(loss)
            evaluation_metrics_list['accuracies'].append(accuracy)

        if epoch % 10 == 0:
            save_snapshot(model, optimizer, epoch, snapshot_path, metrics)

    end_time = time.time()
    plot_results(train_losses, evaluation_metrics_list, end_time - start_time, epoch_num)


def eval_model(model, test_data_loaders, loss_function):
    running_loss = 0.0
    running_accuracy = 0.0
    with torch.no_grad():
        for input_ids, start_indexes, end_indexes, labels in test_data_loaders:
            input_ids = input_ids.cuda()
            start_indexes = start_indexes.cuda()
            end_indexes = end_indexes.cuda()
            labels = labels.cuda().squeeze(-1)
            y_hats = model(input_ids, start_indexes, end_indexes)
            loss = loss_function(y_hats, labels.view(-1))
            running_loss += loss.item()
            running_accuracy += calculate_accuracy(y_hats, labels)
    running_loss /= len(test_data_loaders)
    running_accuracy /= len(test_data_loaders)

    return running_accuracy, running_loss


def calculate_accuracy(predictions, labels):
    classes = torch.argmax(predictions, dim=1)
    return torch.sum((classes == labels).int())


def save_snapshot(model, optimizer, epoch, base_path, metrics):
    file_name = f'{epoch}'

    for data_set in metrics:
        file_name += f'_{data_set}-{metrics[data_set]["accuracy"]}-{metrics[data_set]["loss"]}'

    torch.save({
        'epoch': epoch,
        'optimizer_state_dict': optimizer.state_dict(),
        'model_state_dict': model.state_dict(),
        'metrics': metrics
    })


def plot_results(train_metrics, evaluation_metrics, execution_time, number_of_epochs):
    execution_time_in_hours = execution_time / 3600
    train_x = linspace(0, number_of_epochs, len(train_metrics), endpoint=True)
    plt.plot(train_x, train_metrics, color='red')
    plt.title('loss over epochs')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.figtext(0.5, 0.01, "OK", ha='total execution time {:.3f}h'.format(execution_time_in_hours))
    plt.savefig('train_results.png')
    plt.clf()

    for dataset in evaluation_metrics:
        losses = evaluation_metrics[dataset]['losses']
        accuracies = evaluation_metrics[dataset]['accuracies']
        test_x = linspace(0, number_of_epochs, len(accuracies), endpoint=True)
        plt.plot(test_x, losses, color='blue', label='Loss over epochs')
        plt.plot(test_x, accuracies, color='green', label='Accuracy over epochs')
        plt.xlabel('epochs')
        plt.title(f'Test Results For {dataset}')
        plt.figtext(0.5, 0.01, "OK", ha='total execution time {:.3f}h'.format(execution_time_in_hours))
        plt.legend()
        plt.savefig(f'{dataset}-test-results.png')
        plt.clf()
