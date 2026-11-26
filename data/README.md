# Data (not committed)

The SentiMix / SemEval-2020 Task 9 **Hinglish** data is public and is **not redistributed** in this
repository (`data/*.txt` is git-ignored). Obtain it from the official release:

- Zenodo record 3974927: <https://zenodo.org/record/3974927> (`Semeval_2020_task9_data.zip`, 2.8 MB,
  md5 `4d93553c213b3a966b6d55b7a7c03be7`)
- Task page: <https://ritual-uh.github.io/sentimix2020/>

Extract the **Hinglish** files and place exactly these four files in this folder (`data/`):

| File | Content |
|---|---|
| `Hinglish_train_14k_split_conll.txt` | 14,000 labelled training tweets (CoNLL, one token per line) |
| `Hinglish_dev_3k_split_conll.txt` | 3,000 labelled development tweets |
| `Hinglish_test_unalbelled_conll_updated.txt` | 3,000 test tweets (labels removed) |
| `Hinglish_test_labels.txt` | CSV `Uid,Sentiment` with the 3,000 test labels |

(The misspelling `unalbelled` is in the original file name and is what `src/load_data.py` expects.)

## Verify

```bash
python src/load_data.py
# expected: 14000 3000 3000, neutral 7492, positive 6616, negative 5892
```

SHA-256 of the files used for every number in the report (these copies have CRLF line endings;
the loader works with either CRLF or LF, but the checksums below only match CRLF copies):

```
b2bc9e9e2494693ab3b0fd025ffb0292a3c4ab959861306dd1d993fa7e7c2177  Hinglish_train_14k_split_conll.txt
9679768e0b6697be4c492b41f876756023fdb0face8cb86f8cd6cdb767acd1dc  Hinglish_dev_3k_split_conll.txt
b71df6f50421d78a5c44df02d9e1dee537c3e758bf424dddf420c3ef59b00bdf  Hinglish_test_unalbelled_conll_updated.txt
adc2317d11bd55b8216fe1c7479bd04020fbbbb96764e59e3c7156423a73d077  Hinglish_test_labels.txt
```

Citation: Patwa et al., *SemEval-2020 Task 9: Overview of Sentiment Analysis of Code-Mixed Tweets*,
SemEval 2020, pp. 774-790.
