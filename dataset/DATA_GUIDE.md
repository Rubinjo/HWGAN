## Dataset guidelines

### IAM (currently not supported)

1. Register for free at this [website](http://www.fki.inf.unibe.ch/databases/iam-handwriting-database).
2. Download `words.tgz` and `ascii.tgz`.
3. Create the directory `dataset/IAM/words/`.
4. Put `ascii.tgz/words.txt` into `dataset/IAM/`.
5. Put the content (directories `a01`, `a02`, ...) of `words.tgz` into `dataset/IAM/words/`.

### EMNIST

1. Download EMNIST (by_class) dataset at this [website](https://www.nist.gov/srd/nist-special-database-19).
2. Create the directory `dataset/EMNIST/`.
3. Put directories `4a`, `4b`, ... into `dataset/EMNIST/`.
4. Only the .png files are required, so all other files can be deleted, but this is not necessary.
