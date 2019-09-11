import os, codecs, pandas as pd


def make_pic_from_mu2(fp):
    print(fp)
    doc = codecs.open(os.path.abspath(fp), 'rU', 'UTF-8')
    df = pd.read_csv(doc, sep='\t')


def main():
    dp = os.path.join(os.environ['HOMEPATH'], 'Desktop', 'SymbTr-master', 'mu2')
    fp = os.path.join(dp, 'hicaz--turku--nimsofyan--daglar_daglar--.mu2')
    make_pic_from_mu2(fp)


if __name__ == '__main__':
    main()
