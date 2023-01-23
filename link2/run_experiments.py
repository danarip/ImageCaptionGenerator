from datetime import datetime
from link2.ImageCaptioningUnified import single_run
from definitions import cwd


def oredered_dict_to_line(res):
    terms = list()
    for key, value in res.items():
        terms.append(str(value))
    line = ",".join(terms)
    return line


def save_str_to_file(path, s):
    text_file = open(path, "w")
    n = text_file.write(s)
    text_file.close()


def run_experiments():
    exp_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = f"{cwd}/results/exp_{exp_id}.txt"
    data_limit = None
    num_epochs = 40
    batch_size = 256

    lines = list()
    res = single_run(run_mode="transformer", num_epochs=num_epochs, data_limit=data_limit, batch_size=batch_size,
                     d_model=200)
    lines.append(oredered_dict_to_line(res))
    save_str_to_file(path, "\n".join(lines))

    res = single_run(run_mode="lstm", num_epochs=num_epochs, data_limit=data_limit, batch_size=batch_size,
                     decoder_dim=200)
    lines.append(oredered_dict_to_line(res))
    save_str_to_file(path, "\n".join(lines))

    res = single_run(run_mode="transformer", num_epochs=num_epochs, data_limit=data_limit, batch_size=batch_size,
                     d_model=128)
    lines.append(oredered_dict_to_line(res))
    save_str_to_file(path, "\n".join(lines))

    res = single_run(run_mode="transformer", num_epochs=num_epochs, data_limit=data_limit, batch_size=batch_size,
                     d_model=256)
    lines.append(oredered_dict_to_line(res))
    save_str_to_file(path, "\n".join(lines))

    res = single_run(run_mode="transformer", num_epochs=num_epochs, data_limit=data_limit, batch_size=batch_size,
                     d_model=512)
    lines.append(oredered_dict_to_line(res))
    save_str_to_file(path, "\n".join(lines))

    res = single_run(run_mode="lstm", num_epochs=num_epochs, data_limit=data_limit, batch_size=batch_size,
                     decoder_dim=128)
    lines.append(oredered_dict_to_line(res))
    save_str_to_file(path, "\n".join(lines))

    res = single_run(run_mode="lstm", num_epochs=num_epochs, data_limit=data_limit, batch_size=batch_size,
                     decoder_dim=256)
    lines.append(oredered_dict_to_line(res))
    save_str_to_file(path, "\n".join(lines))

    res = single_run(run_mode="lstm", num_epochs=num_epochs, data_limit=data_limit, batch_size=batch_size,
                     decoder_dim=512)
    lines.append(oredered_dict_to_line(res))
    save_str_to_file(path, "\n".join(lines))

    res = single_run(run_mode="lstm", num_epochs=num_epochs, data_limit=data_limit, batch_size=batch_size,
                     decoder_dim=1024)
    lines.append(oredered_dict_to_line(res))
    save_str_to_file(path, "\n".join(lines))

if __name__ == "__main__":
    run_experiments()
