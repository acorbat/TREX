Installation
------------

This makes the dependencies available:

    python3 -m venv venv
    venv/bin/pip install -e .
    source venv/bin/activate


Minimal test dataset
--------------------

`mini/` contains a small test dataset. The BAM file was created by downsampling
a full dataset:

    samtools view -s 0.1 -b /proj/uppstore2018019/P10306/pMR439_clone16_P10306_1002/10x_14_pMR439_clone16/outs/possorted_genome_bam.bam chrTomato-N > new.bam

`barcodes.tsv` is a copy of `/proj/uppstore2018019/P10306/pMR439_clone16_P10306_1002/10x_14_pMR439_clone16/outs/filtered_gene_bc_matrices/hg38_Tomato-N/barcodes.tsv`

`chrTomato-N.fa` is a copy of `/proj/uppstore2018019/hg38_tdTomato-N/chrTomato-N.fa`,
with an `A` appended at the end of each line to counter the problem that the
original file had DOS line breaks, which STAR interpreted as extra nucleotides.


Running the tool
----------------

On the minimal test dataset:

    rm -r lineage_run/
    braintrace -s 695 -e 724 mini/
