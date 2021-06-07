import scipy, numpy, shutil, os, nibabel, re
import sys, getopt

import imageio

def convert(inputfile, outputfile, image_name_root, mode = 'ma'):

    # set fn as your 4d nifti file
    image_array = nibabel.load(inputfile).get_data()
    print(len(image_array.shape))

    # if 4D image inputted
    if len(image_array.shape) == 4:
        # set 4d array dimension values
        nx, ny, nz, nw = image_array.shape

        # set destination folder
        if not os.path.exists(outputfile):
            os.makedirs(outputfile)
            print("Created ouput directory: " + outputfile)

        print('Reading NIfTI file...')

        total_volumes = image_array.shape[3]
        total_slices = image_array.shape[2]

        # iterate through volumes
        for current_volume in range(0, total_volumes):
            slice_counter = 0
            # iterate through slices
            for current_slice in range(0, total_slices):
                if (slice_counter % 1) == 0:
                    # rotate or no rotate
                    data = image_array[:, :, current_slice, current_volume] *255
                    # alternate slices and save as png
                    print('Saving image' + str(slice_counter))
                    image_name = image_name_root + "%5d"%(slice_counter) + ".png"
                    imageio.imwrite(image_name, data)
                    print('Saved.')

                    # move images to folder
                    print('Moving files...' + str(slice_counter))
                    src = image_name
                    shutil.move(src, outputfile)
                    slice_counter += 1
                    print('Moved.'+ str(slice_counter))

        print('Finished converting images')

        return 

    # else if 3D image inputted
    elif len(image_array.shape) == 3:
        # set 4d array dimension values
        nx, ny, nz = image_array.shape

        # set destination folder
        if not os.path.exists(outputfile):
            os.makedirs(outputfile)
            print("Created ouput directory: " + outputfile)

        print('Reading NIfTI file...')

        total_slices = image_array.shape[2]

        slice_counter = 0
        # iterate through slices
        for current_slice in range(0, total_slices):
            # alternate slices
            if (slice_counter % 1) == 0:
                data = image_array[:, :, current_slice]
                data = data.astype('uint8').T
                if mode == 'ma':
                    for i in range(nx):
                        for j in range(ny):
                            if data[i,j]==1:
                                data[i,j] = data[i,j]*255
                            if data[i,j] == 2:
                                data[i,j] = data[i,j]*127.5
                else:
                    for i in range(nx):
                        for j in range(ny):
                            if data[i,j]==1:
                                data[i,j] = data[i,j]*255
                            if data[i,j] == 2:
                                data[i,j] = 0

                # alternate slices and save as png
                if (slice_counter % 1) == 0:
                    print('Saving image...'+ str(slice_counter))
                    image_name = image_name_root + "%05d"%(slice_counter+1) + ".jpg"
                    imageio.imwrite(image_name, data)
                    print('Saved.'+ str(slice_counter))

                    # move images to folder
                    print('Moving image...'+ str(slice_counter))
                    src = image_name
                    shutil.move(src, outputfile)
                    slice_counter += 1
                    print('Moved.')

        print('Finished converting images')
        return

    else:
        print('Not a 3D or 4D Image. Please try again.')
        return

def main(argv):
    inputfolder = 'Desktop/ScrData'
    outputfolder = 'Desktop/ScrData/IMG'
    mode ='ma'
    labelFolder = 'Label'
    sampleFolder = 'Sample'
    try:
        opts, args = getopt.getopt(argv, "i:o:", ["ifile=", "ofile="])
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
    

    print('Input folder is ', inputfolder)
    print('Output folder is ', outputfolder)
    
    labelPath = inputfolder + '/' + labelFolder
    labelFiles = os.listdir(labelPath)
    
    for file in labelFiles:
        image_root = re.match('mask-*[0-9]*[0-9]',file)
        image_root = image_root.group()  
        convert(labelPath + '/' + file, outputfolder + '/' + 'Label_ma',image_root, mode)
    
    samplePath = inputfolder + '/' + sampleFolder
    sampleFiles = os.listdir(samplePath)

    for file in sampleFiles:
        image_root = re.match('sample[0-9]*[0-9]',file)
        image_root = image_root.group()  
        convert(labelPath + '/' + file, outputfolder + '/' + 'sample',image_root, mode)

# call the function to start the program
if __name__ == "__main__":
    main(sys.argv[1:])