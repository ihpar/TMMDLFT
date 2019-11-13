from nc_dictionary import NCDictionary
from oh_manager import OhManager
from model_ops import load_model


def create_training_data(makam, model_a, model_b):
    with open(makam + '--nc_corpus.txt', 'r') as crp:
        songs = crp.read().splitlines()
        for song in songs:
            notes = song.split(' ')
            for nd in notes:
                pass


def main():
    makam = 'hicaz'
    dir_path = 'C:\\Users\\istir\\Desktop\\SymbTr-master\\mu2'
    note_dict = NCDictionary()
    oh_manager = OhManager(makam)

    model_a = load_model(makam, 'sec_AW6_v61')
    model_b = load_model(makam, 'sec_AW7_v62')

    create_training_data(makam, model_a, model_b)


if __name__ == '__main__':
    main()
