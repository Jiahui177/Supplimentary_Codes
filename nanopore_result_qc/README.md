# Nanopore suquencing result quailty control -Section 2.3.3
This folder contains the scripts mentioned in Section 2.3.3 -Yeast library screening and nanopore sequencing-based binder identification.
QC.py filters Oxford Nanopore FASTQ reads by length and (optionally) by mean quality before alignment to the binder reference sequences.
## Filtering used in this thesis:
```
python QC.py \
  --input  barcode01.fastq.gz \
  --min-length 250 \
  --max-length 700 \
  --min-mean-q 15 \
  --write-failed
```
