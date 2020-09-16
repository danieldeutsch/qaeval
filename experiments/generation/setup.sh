DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

# Download the data from the CodaLab worksheet
mkdir -p ${DIR}/data
wget https://worksheets.codalab.org/rest/bundles/0x36403eba6daf46acbc5729cb1680a001/contents/blob/ -O ${DIR}/data/train.tsv
wget https://worksheets.codalab.org/rest/bundles/0x303c441ba0d04062a293a4b83c86af77/contents/blob/ -O ${DIR}/data/dev.tsv
wget https://worksheets.codalab.org/rest/bundles/0x2a42519198824d9bbb60bbba0fe629b6/contents/blob/ -O ${DIR}/data/combined_neg_pos.tsv

# Reformat
python ${DIR}/preprocess.py ${DIR}/data/train.tsv ${DIR}/data/train.jsonl
python ${DIR}/preprocess.py ${DIR}/data/dev.tsv ${DIR}/data/valid.jsonl
python ${DIR}/preprocess.py ${DIR}/data/combined_neg_pos.tsv ${DIR}/data/combined_neg_pos.jsonl
