from Bio import Entrez, SeqIO

Entrez.email = "lahed97318@exemplo.com"
handle = Entrez.efetch(db="nucleotide", id="NC_045512.2", rettype="fasta", retmode="text")
record = SeqIO.read(handle, "fasta")
SeqIO.write(record, "wuhan_reference.fasta", "fasta")
handle.close()
