import nibabel as nb
import numpy as np
import sys, getopt

def main(Args):
    inputfolder = '../SrcData'
    outputfolder = '../SrcData/IMG'
    labelFolder = 'Label'
    sampleFolder = 'Sample'
    try:
        opts, args = getopt.getopt(Args, "i:o:", ["ifile=", "ofile="])
    except getopt.GetoptError:
        print('nii2png.py -i <inputfolder> -o <outputfolder>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('nii2png.py -i <inputfolder> -o <outputfolder>')
            sys.exit()
        elif opt in ("-i", "--input"):
            inputfolder = arg
        elif opt in ("-o", "--output"):
            outputfolder = arg
