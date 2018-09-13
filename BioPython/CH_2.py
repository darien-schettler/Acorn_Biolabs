from Bio.Seq import Seq
from Bio import SeqIO

"""
------------------------------
 2.2 - WORKING WITH SEQUENCES
------------------------------

Disputably (of course!), the central object in bioinformatics is the sequence. 

Thus, we’ll start with a quick introduction to the Biopython mechanisms for dealing with sequences, the Seq object

Most of the time when we think about sequences we have in my mind a string of letters like ‘AGTACACTGGT’

You can create such Seq object with this sequence as follows:

"""

my_seq = Seq('AGTACACTGGT')  # Create a 'SEQ' object using the Seq command imported through Bio

print(type(my_seq))
print(my_seq)  # Print the sequence (should appear similar to a string)

print()

print(my_seq.alphabet)
# What we have here is a *generic* alphabet
# This reflects the fact that we have not specified if the sequence is a protein or DNA sequence
# In addition to having an alphabet, the Seq object differs from Python's string type in the methods it supports

print("\nLet's see some of the methods associated with the Seq object\n")
print("This is the original sequence for reference  --->  {}".format(my_seq))
print("This is the complement sequence: --------------->  {}".format(my_seq.complement()))
print("This is the reverse complement sequence  ------->  {}\n".format(my_seq.reverse_complement()))

# The next most important class is the SeqRecord or Sequence Record
# This holds a sequence (as a Seq object) with additional annotation including:
#   1. Identifier
#   2. Name
#   3. Description

# The Bio.SeqIO module for reading and writing sequence file formats works with SeqRecord and will be introduced below


"""
-----------------------
 2.3 - A Usage Example
-----------------------

Before we jump right into parsers and everything else to do with Biopython, let’s set up an example...

Having just completed a recent trip, we’ve suddenly developed an interest with Lady Slipper Orchids

Orchids are beautiful and extremely interesting for people studying evolution and systematics

So let’s suppose we’re thinking about writing a funding proposal to do a molecular study of Lady Slipper evolution
    -- We want to discover the kind of research that has already been done and how we can add to that

After a little bit of reading up we discover: 
    1. The Lady Slipper Orchids are in the Orchidaceae family
    2. The Lady Slipper Orchids are in the Cypripedioideae sub-family 
    3. The Lady Slipper Orchids are made up of 5 genera: 
        a. Cypripedium
        b. Paphiopedilum
        c. Phragmipedium
        d. Selenipedium
        e. Mexipedium

That gives us enough to get started delving for more information

So, let’s look at how the Biopython tools can help us

"""

"""
-------------------------------------
 2.4 - Parsing Sequence File Formats
-------------------------------------

A large part of much bioinformatics work is dealing with the many types of file formats designed to hold biological data

These files are loaded with interesting biological data
It's a special challenge to parse these files into a format so that you can manipulate them using a programming language

However the task of parsing these files can be frustrated by the fact that the formats can change quite regularly
    -- Also, formats may contain small subtleties which can break even the most well designed parsers


We are now going to briefly introduce the Bio.SeqIO module – you can find out more in Chapter 5


We’ll start with an online search for our friends, the lady slipper orchids

To keep this introduction simple, we’re just using the NCBI website by hand

In the nucleotide databases at NCBI, we use an Entrez online search for everything mentioning the text Cypripedioideae
    -- http://www.ncbi.nlm.nih.gov:80/entrez/query.fcgi?db=Nucleotide

When this tutorial was originally written, this search gave us only 94 hits
    - We saved these as a FASTA formatted text file and as a GenBank formatted text file
        --> files ls_orchid.fasta
        --> ls_orchid.gbk
                                    <<< Included with the Biopython source code under docs/tutorial/examples/ >>>

If you run the search today, you’ll get hundreds of results!

When following the tutorial, if you want to see the same list of genes, just download the two files above

In Section 2.5 we will look at how to do a search like this from within Python

"""

# Let's parse the fasta file using SeqIO
for seq_record in SeqIO.parse("ls_orchid.fasta", "fasta"):
    print(seq_record.id)
    print(repr(seq_record.seq))
    print(len(seq_record))

# Let's parse the gbq file using SeqIO
for seq_record in SeqIO.parse("ls_orchid.gbk", "genbank"):
    print(seq_record.id)
    print(repr(seq_record.seq))
    print(len(seq_record))

"""
--------------------------------------------
 2.5 - Connecting With Biological Databases
--------------------------------------------

One of the very common things that you need to do in bioinformatics is extract information from biological databases

It can be quite tedious to access these databases manually, especially if you have a lot of repetitive work to do

Biopython attempts to save you time and energy by making some on-line databases available from Python scripts


Currently, Biopython has code to extract information from the following databases:
    1. Entrez (and PubMed) from the NCBI – See Chapter 9
    2. ExPASy – See Chapter 10
    3. SCOP – See the Bio.SCOP.search() function
    
The code in these modules basically makes it easy to write Python code that interact with the CGI scripts on these pages
    -- This is so that you can get results in an easy to deal with format
    -- In some cases, the results can be integrated with the Biopython parsers to make it easier to extract information
"""

"""
-----------------------
 2.6 - What To Do Next
-----------------------

Now that you’ve made it this far, you hopefully have a good understanding of the basics of Biopython
We're ready to start using it for doing useful work

Now we will move on to the sequence objects chapter (3)

"""