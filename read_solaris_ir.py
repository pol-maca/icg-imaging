#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import datetime
import errno
import io
import json
import os
import warnings
from typing import Any, Union

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from PIL import Image, ImageOps
from matplotlib import cm
from matplotlib.ticker import FixedLocator, FormatStrFormatter

warnings.filterwarnings(action='once')

# Options for plotting
large = 22
med = 16
small = 12
params = {'legend.fontsize': med,
          'figure.figsize': (16, 10),
          'axes.labelsize': med,
          'axes.titlesize': med,
          'xtick.labelsize': med,
          'ytick.labelsize': med,
          'figure.titlesize': large}
plt.rcParams.update(params)

plt.style.use('seaborn-dark')
sns.set_style("white")


def show(msg):
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    num_dashes = max(80, len(msg) + len(now) + 1)
    print("-" * num_dashes)
    print(now + " " + msg)
    print("-" * num_dashes)


# Helper function to walk directory structure taking only directories
def get_immediate_subdirectories(a_dir):
    return [name for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))]


# Helper to silently remove files
def quiet_remove(filename):
    try:
        os.remove(filename)
    except OSError as e:
        # errno.ENOENT = no such file or directory
        if e.errno != errno.ENOENT:
            # re-raise exception if a different error occurred
            raise


# Metadata files specify which channels were used for imaging
# This dictionary is used to conver the channel number to
# a readable format used in the file naming
channels = {
    '1': '470',
    '2': '660',
    '3': '750',
    '4': '800',
    '5': '000'  # Error or unknown channel
}
# The file extensions indicate which type of file
# each file consists of 1 second of video from
# stacked images at 30 fps - so 30 frames in total
image_types = {
    'svr': 'RGB',
    'sva': 'Monochrome',
    'svm': 'Side-by-Side'
}


def get_channel(full_dir):
    """
    Read the metadata file in a video folder to extract the wavelength setting.
    :param full_dir: driectory containing a metadata.svd JSON file
    :return: integer value of the NIR wavelength (called 'Channel' in the metadata)
    """
    input_file = os.path.join(full_dir, 'metadata.svd')
    with open(input_file, 'rb') as f:
        metadata = f.read()
    as_json = metadata.decode('utf8').replace("'", '"').replace('"" ', ' ')
    data = json.loads(as_json)
    return int(data['Channel'])


def get_wl(full_dir):
    """
    Get the NIR wavelength used for this Video session.
    :param full_dir: directory containing SVM/SVR files
    :return: a string containing the wavelength
    """
    return channels[str(get_channel(full_dir))]


def next_ir_frame(full_dir):
    """
    SVM - Monochrome Videos for fluorescence side-by-side video

    Structure of the file:
     header_svm - 32 byte Global Header for the file

     Per image: (There are 30 images in each SVM file)
     file_header_svm  - 48 byte Image Header
     frame_len_svm - 1024 x 1040 x 2 bytes
            Each frame_len_svm contains a 1024 x 1024 byte image plus a 16 byte order so
            we set 'trim  = 16' to clip it during processing.

    Allocating space for 2 images (plus borders) seems to be related to side-by-side display.
    However we simply take the first of the images as we only require the NIR intensity
    readings.

    :returns
    A monochrome image containing the NIR intensities
    """

    '''
    The next variable define the SVM file layout. 
    Do not change them unless you are sure you understand the consequences.
    '''
    header_svm: int = 32
    file_header_svm: int = 48
    shape_svm = (1024, 1040)
    frame_len_svm = shape_svm[0] * shape_svm[1] * 2
    trim: int = 16
    ''' End of SVM file layout variables. '''

    svm_files: list[Union[Union[str, bytes], Any]] = []
    for _, _, files in os.walk(full_dir):
        svm_files = sorted([f for f in files if f.endswith(".svm")])

    for svm_file in svm_files:
        f = os.path.join(full_dir, svm_file)
        with open(f, 'rb') as fb:
            buffer = fb.read()
        bytes_read = os.stat(f).st_size

        x1 = header_svm + frame_len_svm
        x2 = frame_len_svm + x1
        with open('example.svm', 'wb') as ex:
            ex.write(buffer[:header_svm + 2*(file_header_svm+frame_len_svm)])

        while x2 <= bytes_read:
            byte_array = np.frombuffer(buffer[x1:x2], dtype=np.uint16)
            raw = np.reshape(byte_array, shape_svm)

            x1 = x2 + (2 * file_header_svm) + frame_len_svm
            x2 = x1 + frame_len_svm

            im = Image.fromarray(np.int32(raw[:, trim:]), mode='I')
            im = im.rotate(90)
            im = ImageOps.flip(im)

            yield im


def next_rgb_frame(full_dir):
    """
    Get next RGB image from SVR file
    SVR - RGB videos of white light images

    Parameters:
        full_dir - directory containing valid svr files

    Returns
        PIL Image in RGB format

    Structure of the file:
     header_svr - 32 byte Global Header for the file

     Per image: (There are 15 images in each SVR file, so you should see twice as many SVM files as SVR files)
     file_header_svr  - 48 byte Image Header
     frame_len_svr - 1024 x 1024 x 3 bytes

    Allocating space for 2 images (plus borders) seems to be related to side-by-side display.
    However we simply take the first of the images as we only require the NIR intensity
    readings.
    """

    '''
    The next variables define the byte layout of the SVR file. 
    Don't change these unless you are confident you understand the consequences.
    '''
    header_svr: int = 32
    file_header_svr: int = 48
    shape_svr = (1024, 1024, 3)
    frame_len_svr = shape_svr[0] * shape_svr[1] * shape_svr[2]
    ''' End of SVR file layout variables. '''

    svr_files: list[Union[Union[str, bytes], Any]] = []
    for _, _, files in os.walk(full_dir):
        svr_files = sorted([f for f in files if f.endswith(".svr")])

    for svr_file in svr_files:
        f = os.path.join(full_dir, svr_file)
        with open(f, 'rb') as fb:
            buffer = fb.read()
        bytes_read = os.stat(f).st_size

        x1: int = header_svr + file_header_svr
        x2: int = x1 + frame_len_svr

        with open('example.svr', 'wb') as ex:
            ex.write(buffer[:header_svr + 2*(file_header_svr+frame_len_svr)])

        while x2 <= bytes_read:
            byte_array = np.frombuffer(buffer[x1:x2], dtype=np.uint8)
            rgb_in = np.reshape(byte_array, shape_svr)
            im = Image.fromarray(rgb_in)
            im = im.rotate(90)
            im = ImageOps.flip(im)

            x1 = x2 + file_header_svr
            x2 = x1 + frame_len_svr

            yield im


def make_plot(z, figsize=(20, 20), scale=255 * 257,
              wavelength=800, terrain=None,
              nir_min=0.2, offset=3.5):
    """
    Make a 3-D plot of image intensity as z-axis and RGB image as an underlay on the z=0 plane.
    :param z: NIR intensities
    :param figsize: size of the figure to default (20,20)
    :param scale: Scale to resize intensities for aesthetics (make intensities <= 1)
    :param wavelength: The wavelength to include in the legend
    :param terrain: The rgb image to include as the x-y plane (default is no underlay)
    :param nir_min: Cutoff for the minimum level of NIR intensity (0 - 1) so the plot is cleaner
    :param offset: Shift the RGB underlay by this amount for visual appeal so there is a space
    :return: a PIL Image
    """
    fig = plt.figure(figsize=figsize)

    ax = fig.gca(projection='3d')

    z = np.float32(bw)

    Z = z / scale

    X, Y = np.arange(0, z.shape[1], 1), np.arange(0, z.shape[0], 1)
    X, Y = np.meshgrid(X, Y)

    surf = ax.plot_surface(X, Y, Z,
                           rstride=3, cstride=3,
                           cmap=cm.coolwarm,
                           alpha=0.3,
                           linewidth=0,
                           antialiased=False,
                           vmin=0, vmax=1)

    if terrain is not None:
        ax.plot_surface(X, Y, -offset * np.ones_like(Z, dtype=np.float32),
                        rstride=5, cstride=5, facecolors=terrain / 255)

        ''' Now overlay the fluorescence '''
        z_fluorescence = Z.copy()
        z_fluorescence[z_fluorescence < nir_min] = 0
        z_rgba = np.ones((Z.shape[0], Z.shape[1], 4))
        z_rgba[:, :, 3] = z_fluorescence[:, :]
        ax.plot_surface(X, Y, -offset * np.ones_like(Z, dtype=np.float32),
                        rstride=5, cstride=5, facecolors=z_rgba)

        ax.set_zlim(-offset, 1)
    else:
        ax.plot_surface(X, Y, (Z / 257) - offset,
                        rstride=3, cstride=3,
                        cmap=cm.coolwarm,
                        alpha=0.4,
                        linewidth=0,
                        antialiased=False,
                        vmin=-offset, vmax=(1.0 / 257.0) - offset)
        ax.set_zlim(-offset, 1)

    ax.zaxis.set_major_locator(FixedLocator([0.0, 0.5, 1.0]))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    ax.set_xlabel('\nHeight (Pixels)')
    ax.set_ylabel('\nWidth (Pixels)')
    ax.set_zlabel('\nRelative NIR\nIntensity')

    ax.view_init(azim=30)

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.4, aspect=5, orientation="horizontal",
                 label='Relative Intensity of fluorescence \nat wavelength ' + r"$\lambda_{f} =$" + "{}nm".format(
                     wavelength))

    buf = io.BytesIO()
    plt.tight_layout(h_pad=1)
    plt.savefig(buf, format='png')
    buf.seek(0)
    img = Image.open(buf)
    plt.close('all')

    return img


if __name__ == "__main__":

    ''''
    The following variables are hard-coded default locations and search terms so we can
    find SVM and SVR files.
    
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
    
    You will need to edit them to suit your project.
    '''
    p_search_term = 'spe_no'  # P-E archive
    v_search_term = 'Video'
    input_dir = 'input'
    out_dir = 'output'

    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)

    if not os.path.isdir(input_dir):
        show(f"Missing or invalid input directory: {input_dir}. Creating empty input directory.")
        os.mkdir(input_dir)

    projects = [p for p in get_immediate_subdirectories(input_dir)
                if p_search_term in p]

    if len(projects) > 0:
        show(f"Found {len(projects)} projects in input folder '{input_dir}'.")
    else:
        show(f"No projects were found in input folder '{input_dir}'. Please check configuration.")

    for project in projects:
        show(f"Processing project: {project}.")
        project_path = os.path.join(input_dir, project)

        video_dirs = [v for v in get_immediate_subdirectories(project_path)
                      if v_search_term in v]

        p_out_dir = os.path.join(out_dir, project)
        if not os.path.isdir(p_out_dir):
            os.mkdir(p_out_dir)

        show("Processing data in the following locations: \n\t{}".format(video_dirs))

        for video_dir in video_dirs:

            show(f"Processing video files in directory: {video_dir}.")

            full_out_dir = os.path.join(p_out_dir, video_dir)
            if not os.path.isdir(full_out_dir):
                os.mkdir(full_out_dir)

            full_in_dir = os.path.join(project_path, video_dir)

            ctr = 0
            ''' 
            As the number of images grows the memory required to hold them grows accordingly.
            So we process each image in turn and save it to the output directory as a sequentially
            numbered PNG file. You can run ffmpeg (for example) to gather the resulting images 
            into a video.
            '''
            for bw, rgb in zip(next_ir_frame(full_in_dir), next_rgb_frame(full_in_dir)):
                ''' Create the stem filename for saving images '''
                f_prefix = '{}'.format(ctr).zfill(8)
                f_stem = os.path.join(full_out_dir, f_prefix)

                '''
                Plot the data as a 3-D plot using the RGB image as the underlay on the z=0
                plane and taking NIR intensity as the z-axis dimension
                '''
                show(f"Processing plot number: {f_prefix}")
                z = np.float32(bw)  # The NIR data
                underlay = np.float32(rgb)  # The RGB data

                ''' Save RGB and BW images in case we need them later'''
                bw.save(f_stem + "_bw.png")
                rgb.save(f_stem + "_rgb.png")

                '''
                Get the NIR wavelength setting for these images. The stack is multi-wavelength
                and so we show the current wavelength in the legend of the 3-D plot
                '''
                wl = get_wl(full_in_dir)

                plot = make_plot(z, figsize=(15, 15), scale=255 * 257,
                                 wavelength=wl, terrain=underlay)

                plot.save(f_stem + "_plot.png")

                ctr += 1

    show("All done. Bye!")
