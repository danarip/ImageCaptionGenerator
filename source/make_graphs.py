import matplotlib
import matplotlib.pyplot as plt

from definitions import cwd
from source.utils import load_file


def create_loss_vs_size(
        path="long_experiment_results/exp_20230121_231314.txt"
):
    full_path = f"{cwd}/{path}"
    s = load_file(full_path)
    lines = s.strip().split("\n")
    lines = filter(lambda line: line[0] != "#", lines)
    terms = [line.split(",", 9) for line in lines]
    typ = list()
    run_time = list()
    test_loss = list()
    val_loss = list()
    memory_size = list()
    num_params = list()
    for line in terms:
        typ.append(line[0] == "lstm")
        num_params.append(float(line[2]))
        run_time.append(float(line[3]))
        test_loss.append(float(line[6]))
        val_loss.append(float(line[5]))
        memory_size.append(float(line[7]))

    idx_lstm = [i for i, x in enumerate(typ) if x]
    idx_transformer = [i for i, x in enumerate(typ) if not x]

    font = {'family': 'normal',
            'weight': 'bold',
            'size': 20}

    matplotlib.rc('font', **font)
    markersize = 15

    lstm_num_params = [num_params[i] for i in idx_lstm]
    transformer_num_params = [num_params[i] for i in idx_transformer]
    lstm_memory_size = [memory_size[i] for i in idx_lstm]
    transformer_memory_size = [memory_size[i] for i in idx_transformer]
    lstm_test_loss = [test_loss[i] for i in idx_lstm]
    transformer_test_loss = [test_loss[i] for i in idx_transformer]
    lstm_run_time_loss = [run_time[i] for i in idx_lstm]
    transformer_run_time_loss = [run_time[i] for i in idx_transformer]

    figure, axis = plt.subplots(1, 2)
    axis[0].plot(lstm_num_params, lstm_test_loss, 'r-*', markersize=markersize, label='lstm')
    axis[0].plot(transformer_num_params, transformer_test_loss, 'b-*', markersize=markersize, label='transformer')
    axis[0].set_xlabel("no. parameters")
    axis[0].set_ylabel("test loss")
    axis[0].legend()

    axis[1].plot(lstm_num_params, lstm_run_time_loss, 'r-*', markersize=markersize, label='lstm')
    axis[1].plot(transformer_num_params, transformer_run_time_loss, 'b-*', markersize=markersize, label='transformer')
    axis[1].set_xlabel("no. parameters")
    axis[1].set_ylabel("training time[sec]")
    axis[1].legend(loc='center right')

    plt.show()


if __name__ == "__main__":
    create_loss_vs_size()
