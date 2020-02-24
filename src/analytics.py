import os


def get_notes(files):
    res = []
    for f in files:
        print(f)
    return res


def plot_weights(files):
    notes = get_notes(files)


def main():
    dir_path = 'C:\\Users\\istir\\Desktop\\SymbTr-master\\mu2'
    files = [f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f)) and f.startswith('hicaz--')]
    plot_weights(files)


if __name__ == '__main__':
    main()
