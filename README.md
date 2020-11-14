# icg-imaging
Supplementary data, Python code and images for the paper _"Dual Color Imaging from a Single BF2-Azadipyrromethene Fluorophore Demonstrated in vivo for Lymph Node Identification" (Niamh Curtin, Dan Wu,Ronan Cahill, Anwesha Sarkar, Pól Mac Aonghusa, Sergiy Zhuk, Manuel Barberio, Mahdi Al-Taher, Jacques Marescaux, Michele Diana, Donal F. O’Shea)_

# Abstract
Dual emissions at ~700 and 800 nm have been achieved from a single NIR-AZA fluorophore 1 by establishing parameters in which it can exist in either its isolated molecular or aggregated states. Dual near infrared (NIR) fluorescence color lymph node (LN) mapping with 1 was achieved in a large-animal porcine model, with injection site, channels and nodes all detectable at both 700 and 800 nm using a preclinical open camera system. The fluorophore was also compatible with imaging using two clinical instruments for fluorescence guided surgery.

# Links to the Paper
Link to [ChemRxiv Preprint](https://chemrxiv.org/articles/preprint/Dual_Color_Imaging_from_a_Single_BF2-Azadipyrromethene_Fluorophore_Demonstrated_in_vivo_for_Lymph_Node_Identification/13203641)

# Code

* This code was written and tested using Python 3.8. The Numpy, Matplotlib, Seaborn and PIL Python packages are required. *

The Python script read_solaris_ir.py contains all of the code necessary to read raw data files exported from the Perkins-Elmer Solaris imaging stack. The script reads NIR & RGB data sources and outputs Black & White NIT images plus Colour RGB images. The images are labelled sequentially as they are read from the raw data. By default the script will save all the images as PNG image files. It will also create the 3D image overlays shown in the paper. The 3D images take most time to compute - so if you do not require them you can comment out line 411:
```Python
    plot = make_plot(z, figsize=(15, 15), scale=255 * 257, 
                     wavelength=wl, terrain=underlay)
```

    The Perkins-Elmer archive has a structure like:
    root
    |_ Project_1
        |_ Video_1
        :
        |_ Video_M
    |
    :
    |_ Project_N
        |_ Video_1
        :
        |_ Video_X
        
    The Project directories are named using the identifier that was entered into the camera when the 
    project was created at the start of the session. In the example provided our projects were named
    'spe_no1, spe_no2, ...' so we use 'spe_no' as the common search string for project directories. I 
    realise this could be programmed more flexibly - but the objective is to get the data from the
    archive - and this is simple and works :-)
    
    The Video_X directories are where the image data are placed. A new Video_X is created every time 
    the NIR wavelength is changed on the camera. 
    
The following variables ```(Lines 338 - 341)``` are hard-coded default locations and search terms so we can find the Solaris SVM and SVR files and save outputs. You should to edit them to suit your project.
```Python
    p_search_term = 'spe_no'  # We use a search term to navigate the raw data, you netered this in the system when you created the project
    v_search_term = 'Video'   # Every time you reset the wavelength on the DSolaris stack a new Video XX directory is created under the project
    input_dir = 'input'       # The root folder where you put the raw data
    out_dir = 'output'        # Where you want the output to go - the script will use the same project/video subdirectory structurefrom the input folder
```
