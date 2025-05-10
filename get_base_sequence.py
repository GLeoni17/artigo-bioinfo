from Bio import Entrez, SeqIO

# Configurações
Entrez.email = "lahed97318@apklamp.com"

def get_base_sequence():
    handle = Entrez.efetch(db="nucleotide", id="NC_045512.2", rettype="fasta", retmode="text")
    record = SeqIO.read(handle, "fasta")
    SeqIO.write(record, "wuhan_reference.fasta", "fasta")
    handle.close()

get_base_sequence()
