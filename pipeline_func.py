"""
Helper pipeline functions for SynSD.

Author: Amandeep Singh Hira
email: ahira1@ualberta.ca
Date: September 2025


References: 

ualbertaIGEM. (2025). Ashbloom https://2025.igem.wiki/ualberta.

test.svg: https://www.svgrepo.com/svg/530662/ribosome

The pandas development team. (2020). pandas [Computer software]. Zenodo. https://doi.org/10.5281/zenodo.3509134

Harris, C. R., Millman, K. J., van der Walt, S. J., Gommers, R., Virtanen, P., Cournapeau, D., Oliphant, T. E. (2020). NumPy [Computer software]. Zenodo. https://doi.org/10.5281/zenodo.4147899

Paszke, A., Gross, S., Massa, F., Lerer, A., Bradbury, J., Chanan, G., ... Chintala, S. (2019). PyTorch [Computer software]. https://pytorch.org/

Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., ... Duchesnay, É. (2011). scikit-learn: Machine learning in Python [Computer software]. Journal of Machine Learning Research, 12, 2825–2830. https://scikit-learn.org/

Streamlit Inc. (2023). Streamlit [Computer software]. https://streamlit.io/

Cock, P. J. A., Antao, T., Chang, J. T., Chapman, B. A., Cox, C. J., Dalke, A., ... de Hoon, M. J. L. (2009). Biopython [Computer software]. Bioinformatics, 25(11), 1422–1423. https://biopython.org/

Hunter, J. D. (2007). Matplotlib: A 2D graphics environment [Computer software]. Computing in Science & Engineering, 9(3), 90–95. https://doi.org/10.1109/MCSE.2007.55

Waskom, M. (2021). Seaborn [Computer software]. https://seaborn.pydata.org/

Lorenz, R., Bernhart, S. H., Höner zu Siederdissen, C., Tafer, H., Flamm, C., Stadler, P. F., & Hofacker, I. L. (2011). ViennaRNA Package [Computer software]. https://www.tbi.univie.ac.at/RNA/

BioRender. (n.d.). BioRender [Computer software]. https://biorender.com/


Description: 
A ribosome binding site (RBS) refers to a brief segment of mRNA that plays a crucial role in attracting the ribosome to start the translation process, thus influencing the efficiency of protein production. For microbial synthetic biology, where meticulous regulation of gene expression is essential, the precise prediction and design of RBS sequences are vital.
Current RBS prediction models predominantly rely on RNA thermodynamics. The most prevalent method is created by the Salis Lab called Denovo DNA [1]. The Denovo DNA estimates RBS sequences by assessing the minimum folding energy of the mRNA alongside the ribosome binding energy, establishing a thermodynamic basis for RBS design [1].
In contrast, we developed a novel deep learning model to predict ribosome binding sites (RBS) and spacer sequences from mRNA contexts. Inspired by natural sequence-to-function mappings, we employ a Transformer encoder–decoder architecture, capable of learning long-range dependencies in RNA. The goal is to enable rational RBS+spacer design for microbial synthetic biology.
The model is coupled with the extraction and analysis code to form a pipeline for easier user interaction. The pipeline takes an annotated genome as an input and extracts out all the ribosome binding sequences and spacer sequences with their respective mRNA sequences. The resulting sequences are used to train the model for sequence prediction. 
The model is validated by calculating the minimum folding energies of the first 25 base pairs of the RNA and the ribosome-RNA sequence binding affinity. As the paper by Chen, Yi-Lan, and Jin-Der Wen elucidates mRNA–ribosome complexes that use less favorable / more structured RBS tend to be disfavored during initiation, via kinetic discrimination [2].

1. Reis, A.C. & Salis, H.M. (2020). An automated model test system for systematic development and improvement of gene expression models. ACS Synthetic Biology, 9(11), 3145-3156.
2. Chen, Yi-Lan, and Jin-Der Wen. "Translation initiation site of mRNA is selected through dynamic interaction with the ribosome." Proceedings of the National Academy of Sciences 119.22 (2022): e2118099119.

"""


import pandas as pd
from Bio import SeqIO
from Bio.Seq import Seq
import io
import os
import RNA

def extract_gene_data(tsv_file, fasta_file, rbs_upstream=25):


    """
    Extract RBS sequence, gene sequence, and gene name from annotated CSV and single genome FASTA.

    Parameters:
    tsv_file : Path to the TSV annotation file
    fasta_file : Path to the genome FASTA file (single sequence)
    rbs_upstream : Number of base pairs upstream to extract for RBS (default: 25)

    Returns:
    df: DataFrame containing gene data
    """

    # Read the single genome sequence from string FASTA DNA file
    genome_record = SeqIO.read(io.StringIO(fasta_file), 'fasta')
    genome_seq = str(genome_record.seq)

    # Read the TSV file
    df = pd.read_csv(io.StringIO(tsv_file), sep='\t')

    # Extract gene data
    gene_data = []
    
    for _, row in df.iterrows():
        begin = int(row['Begin'])
        end = int(row['End'])
        orientation = row['Orientation']
        symbol = row['Symbol'] if pd.notna(row['Symbol']) else ''
        
        # Extract gene sequence using Begin and End as direct pointers (convert to 0-based indexing)
        gene_seq = genome_seq[begin-1:end]
        
        # Extract RBS sequence based on orientation
        if orientation.lower() == 'plus':
            # For plus strand, RBS is upstream of the start position
            rbs_start = max(0, begin - 1 - rbs_upstream)
            rbs_end = begin - 1
            rbs_seq = genome_seq[rbs_start:rbs_end]
        else:  # minus strand
            # For minus strand, RBS is downstream of the end position
            rbs_start = end
            rbs_end = min(len(genome_seq), end + rbs_upstream)
            rbs_seq = genome_seq[rbs_start:rbs_end]
            # Reverse complement for minus strand
            rbs_seq = str(Seq(rbs_seq).reverse_complement())
            gene_seq = str(Seq(gene_seq).reverse_complement())
        
        gene_info = {
            'symbol': symbol,
            'gene_sequence': gene_seq,
            'rbs_sequence': rbs_seq,
        }
        
        gene_data.append(gene_info)



    return pd.DataFrame(gene_data)



def align_rbs_sequences(sequence_a, sequence_b):
    """
    Function that finds the best ungapped local alignment
    by sliding sequence B along sequence A.
    
    Parameters:
    sequence_a (str): The longer reference sequence
    sequence_b (str): The shorter sequence to align
    
    Returns:
    tuple: (aligned_portion, remaining_sequence, alignment_start, alignment_end, matches)
    """
    
    best_matches = 0
    best_start = 0
    seq_b_len = len(sequence_b)
    
    # Slide sequence B along sequence A to find best match
    for i in range(len(sequence_a) - seq_b_len + 1):
        matches = sum(1 for a, b in zip(sequence_a[i:i+seq_b_len], sequence_b) if a == b)
        
        if matches > best_matches:
            best_matches = matches
            best_start = i
    
    alignment_start = best_start
    alignment_end = best_start + seq_b_len
    aligned_portion = sequence_a[alignment_start:alignment_end]
    
    # Get remaining sequence
    after_alignment = sequence_a[alignment_end:]
    remaining_sequence = after_alignment
    
    return aligned_portion, remaining_sequence


def process_sequences_from_df(input_df, output_csv, reference_sequence):
    """
    Read sequences from DataFrame, align them using find_best_ungapped_match,
    and save results to a new CSV file.
    
    Parameters:
    input_df (pd.DataFrame): DataFrame containing sequences
    output_csv (str): Path to output CSV file for results
    reference_sequence (str): Reference sequence to align against

    returns: bool: True if processing and saving was successful, False otherwise and saves data to output_csv
    """
    try:
        # Read input CSV
        df = input_df
        
        # List to store results
        results = []
        
        # Process each sequence
        for index, row in df.iterrows():
            sequence = row['rbs_sequence']  # Assuming 'rbs_sequence' is the column name

            # Apply alignment function
            aligned_portion, remaining = align_rbs_sequences(sequence, reference_sequence)
            
            # Store results
            result = {
                'sequence_id': row.get('id', f'seq_{index}'),
                'gene_name': row.get('gene_name', ''),
                'original_RBS_sequence': sequence,
                'RBS_sequence': aligned_portion,
                'spacer_sequence': remaining,
                'mRNA_sequence': row.get('gene_sequence', ''),
            }
            results.append(result)
        
        # Convert results to DataFrame and save to CSV
        results_df = pd.DataFrame(results)
        results_df.to_csv(output_csv, index=False)
        print(f"Results saved to {output_csv}")
        return True
        
    except Exception as e:
        print(f"Error processing sequences: {e}")
        return False



def anti_sd_converter_for_alignment(sequence):
    """
    Convert DNA sequence to RNA by replacing U with T and making a complementary strand for sequence alignment.
    
    Parameters:
    sequence (str): Input DNA sequence
    
    Returns:
    str: Converted RNA sequence
    """
    rna_sequence = ""
    sequence = sequence.upper()
    for base in sequence:
        if base not in 'ATCGU':
            raise ValueError("Invalid DNA sequence: contains non-ATCGU characters")
        if base == 'A':
            rna_sequence += 'T'
        elif base == 'T' or base == 'U':
            rna_sequence += 'A'
        elif base == 'C':
            rna_sequence += 'G'
        elif base == 'G':
            rna_sequence += 'C'
    rna_sequence = rna_sequence[::-1]
    return rna_sequence



def check_data_files(folder_path):
    """Check files in the specified folder and return a list of file paths if files exist.
    
    args: folder_path (str): The path to the folder to check

    returns: list of file paths (list) or empty list if no files found or error
    """

    # Define the folder path
    temp_folder = folder_path

    # Check if the folder exists (debugging/logging)
    if not os.path.exists(temp_folder):
        print(f"Error: '{temp_folder}' folder does not exist in the current directory.")
        return []
    
    # Get list of files in the folder
    try:
        files = os.listdir(temp_folder)
        
        if not files:
            print(f"The '{temp_folder}' folder is empty.")
            return []
        
        # Collect file paths
        file_paths = []
        
        for file in files:
            file_path = os.path.join(temp_folder, file)
            
            # Check if it's a file (not a directory)
            if os.path.isfile(file_path):
                # Add to file paths list
                file_paths.append(file_path)
        
        
        # Return list of file paths
        return file_paths
        
    except PermissionError:
        print(f"Error: Permission denied accessing '{temp_folder}' folder.")
        return []
    except Exception as e:
        print(f"Error reading '{temp_folder}' folder: {str(e)}")
        return []

anti_SD = "ACCUCCUUA"

def calculate_binding_energy(rna_sequence, anti_sd_sequence=anti_SD):
    """
    Calculate the binding energy between RNA sequence and anti-Shine-Dalgarno sequence using RNAup.
    
    Args:
        rna_sequence (str): The RNA sequence to analyze
        anti_sd_sequence (str): The anti-Shine-Dalgarno sequence (default: ACCUCCUUA)
    
    Returns:
        float: Binding energy in kcal/mol (negative values indicate favorable binding)
    """
    # Create RNAup fold compound for intermolecular structure prediction
    fc = RNA.fold_compound(rna_sequence + "&" + anti_sd_sequence)
    
    # Calculate the minimum free energy of binding
    # RNAup calculates the optimal binding between two RNA sequences
    (structure, energy) = fc.mfe()
    
    return energy


def total_energy(rna_sequence, mrna_sequence, anti_sd_sequence=anti_SD):
    """
    Shows both folding, binding energy and structure in one funcion.

    Args:
        rna_sequence (str): The RNA sequence to analyze
        anti_sd_sequence (str): The anti-Shine-Dalgarno sequence (default: ACCUCCUUA)
    """
    folding_energy_seq = rna_sequence + mrna_sequence
    folding_energy_seq = folding_energy_seq[:25]
    fc = RNA.fold_compound(folding_energy_seq)
    (folding_energy_structure, folding_energy) = fc.mfe()

    be = RNA.fold_compound(rna_sequence + "&" + anti_sd_sequence)
    (binding_energy_structure, binding_energy) = be.mfe()


    plot = RNA.svg_rna_plot(folding_energy_seq, folding_energy_structure, f"temp_data_files/{rna_sequence}.svg")
    return folding_energy, binding_energy
