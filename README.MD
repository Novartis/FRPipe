## README

## about

FRPipe.py is the script used to analyse 19F reporter assay data as described in 
Nature Communications, Hommel et al. "Discovery of a selective and biologically active low-molecular weight antagonist of human IL-1b"

## input data
the script reads preprocessed Bruker files parsing title files enriched with meta data needed for further processing

## title files layout
Ligand: 123 uM LIGANDNAME (LIMS code)  
Protein: 10 uM PROTEINNAME (Protein DB code)  
Reporter: 40 uM REPORTERNAME (LIMS code)

for the reference measurements "Reporter only" and "Reporter with protein" leave out the respective lines.


## Usage

```
# simple run with a reporter KD of 120 uM.
FRPipe.py -i /path/to/bruker/data -t /path/to/where/results/go -k 120 

# -h will provide a detailed description of CLI parameters to tweak the analysis
FRPipe.py -h 

```
