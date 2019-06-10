#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: dataReader.py
# Author: Amir Alansary <amiralansary@gmail.com>

import warnings

warnings.simplefilter("ignore", category=ResourceWarning)
import pandas as pd
import numpy as np
import os
import SimpleITK as sitk
import itk
from tensorpack import logger
from IPython.core.debugger import set_trace

__all__ = [
    "files",
    "filesListFetalUSLandmark",
    "filesListCardioMRLandmark",
    "filesListBrainMRLandmark",
    "NiftiImage",
]

###############################################################################
class files(object):
    """ A class for managing train files

        Attributes:
        directory: input data directo
    """

    def __init__(self, directory=None):

        assert directory, "There is no directory containing training files given"

        self.dir = directory

        # todo make it generic for directories and files with different scenarios
        self.images_list = self._listImages()
        self.landmarks_list = self._listLandmarks()
        self.all_landmarks_list = self._listLandmarks_all()

    def _listImages(self):

        childDirs = listFiles(self.dir, "*")

        image_files = []

        for child in childDirs:
            dir_path = os.path.join(self.dir, child)
            if not (os.path.isdir(dir_path)):
                continue
            # todo: extend to all nifti image extensions
            file_name = listFiles(dir_path, "*.nii.gz")
            file_path = os.path.join(dir_path, file_name[0])
            image_files.append(file_path)

        return image_files

    def _listLandmarks(self):

        childDirs = listFiles(self.dir, "*")
        landmarks = []

        for child in childDirs:
            dir_path = os.path.join(self.dir, child)
            if not (os.path.isdir(dir_path)):
                continue

            file_name = listFiles(dir_path, "*.mps")
            file_path = os.path.join(dir_path, file_name[0])
            points = np.array(extractPointsXML(file_path))
            landmarks.append(np.array(points[:, 2]))

        return landmarks

    def _listLandmarks_all(self):
        # extend directory path
        current_dir = self.dir + "/landmarks"
        childDirs = listFiles(current_dir, "*.txt")
        landmarks = []

        for child in childDirs:
            file_name = os.path.join(current_dir, child)
            file_path = os.path.join(current_dir, file_name)
            points = np.array(extractPointsTXT(file_path))
            landmark = np.array(points)  # all landmark point
            landmarks.append(landmark)

        return landmarks

    def sample_random(self):
        """ return a random sampled ImageRecord from the list of files
        """
        # todo: fix seed for a fair comparison between models
        random_idx = np.random.randint(low=0, high=len(self.images_list))
        sitk_image, image = niftiImage().decode(self.images_list[random_idx])
        landmark = np.array(
            sitk_image.TransformPhysicalPointToIndex(self.landmarks_list[random_idx])
        )
        return image, landmark, random_idx

    def sample_circular(self, shuffle=False):
        """ return a random sampled ImageRecord from the list of files
        """
        if shuffle:
            indexes = rng.choice(x, len(x), replace=False)
        else:
            indexes = np.arange(self.num_files)

        while True:
            for idx in indexes:
                sitk_image, image = niftiImage().decode(self.images_list[idx])
                landmark = np.array(
                    sitk_image.TransformPhysicalPointToIndex(self.landmarks_list[idx])
                )
                image_filename = self.images_list[idx][:-7]
                yield image, landmark, image_filename, sitk_image.GetSpacing()

    @property
    def num_files(self):
        return len(self.images_list)

    # def _get_target_loc(self,filename):
    #     ''' return the center of mass of a given label (target location)
    #     '''
    #     label_image = niftiImage().decode_nifti(self.label_file)
    #     return np.round(center_of_mass(label_image.data))


###############################################################################
#######################################################################
## extract points from vtk file
def extractPointsTXT(filename):
    x = []
    y = []
    z = []
    with open(filename) as f:
        for line in f:
            point = line.split()
            x.append(float(point[0]))
            y.append(float(point[1]))
            z.append(float(point[2]))

    return x, y, z


def getLandmarksFromVTKFile(file):
    """ 0-2 RV insert points
        1 -> RV lateral wall turning point
        3 -> LV lateral wall mid-point
        4 -> apex
        5-> center of the mitral valve
    """
    with open(file) as fp:
        landmarks = []
        for i, line in enumerate(fp):
            if i == 5:
                landmarks.append([float(k) for k in line.split()])
            elif i == 6:
                landmarks.append([float(k) for k in line.split()])
            elif i > 6:
                landmarks = np.asarray(landmarks).reshape((-1, 3))
                landmarks[:, [0, 1]] = -landmarks[:, [0, 1]]
                return landmarks


###############################################################################


class filesListCardioMRLandmark(object):
    """ A class for managing train files for mri cardio data

        Attributes:
        directory: input data directo
    """

    def __init__(self, files_list=None, returnLandmarks=True, agents=2):
        # check if files_list exists
        assert files_list, "There is no directory containing files list"
        self.agents = agents
        # read image filenames
        self.image_files = [line.split("\n")[0] for line in open(files_list[0])]
        # read landmark filenames if task is train or eval
        self.returnLandmarks = returnLandmarks
        if self.returnLandmarks:
            self.landmark_files = [line.split("\n")[0] for line in open(files_list[1])]
            assert len(self.image_files) == len(
                self.landmark_files
            ), "number of image files is not equal to number of landmark files"

    @property
    def num_files(self):
        return len(self.image_files)

    def sample_circular(self, shuffle=False):
        """ return a random sampled ImageRecord from the list of files
        """
        if shuffle:
            indexes = rng.choice(x, len(x), replace=False)
        else:
            indexes = np.arange(self.num_files)

        while True:
            for idx in indexes:
                sitk_image, image = NiftiImage().decode(self.image_files[idx])
                landmarks = []
                images = []
                image_filenames = []
                if self.returnLandmarks:
                    landmark_file = self.landmark_files[idx]
                    all_landmarks = getLandmarksFromVTKFile(landmark_file)
                    # transform landmarks to image coordinates
                    all_landmarks = [
                        sitk_image.TransformPhysicalPointToContinuousIndex(point)
                        for point in all_landmarks
                    ]
                    # 0-2 RV insert points
                    # 1 -> RV lateral wall turning point
                    # 3 -> LV lateral wall mid-point
                    # 4 -> apex
                    # 5-> center of the mitral valve
                    for i in range(0, self.agents):
                        landmarks.append(
                            np.round(all_landmarks[(i + 4) % 6]).astype("int")
                        )

                else:
                    landmarks = None

                # extract filename from path
                for i in range(0, self.agents):
                    images.append(image)
                    image_filenames.append(self.image_files[idx][:-7])

                yield images, landmarks, image_filenames, sitk_image.GetSpacing()


###############################################################################
###############################################################################

######################################################################
## extract points from txt file
def getLandmarksFromTXTFile(file):
    """
    Extract each landmark point line by line and return vector containing all landmarks.
    """
    with open(file) as fp:
        landmarks = []
        for i, line in enumerate(fp):
            landmarks.append([float(k) for k in line.split(",")])
        landmarks = np.asarray(landmarks).reshape((-1, 3))
        return landmarks

indices = ['AC', 'BPons', 'CM', 'LE', 'PC', 'RE', 'RP', 'RP_front', 'SMV', 'VN4', 'callosum_left', 'callosum_right', 'dens_axis', 'genu', 'l_caud_head', 'l_corp', 'l_front_pole', 'l_inner_corpus', 'l_lat_ext', 'l_occ_pole', 'l_prim_ext', 'l_sup_ext', 'l_temp_pole', 'lat_left', 'lat_right', 'lat_ven_left', 'lat_ven_right', 'left_cereb', 'left_lateral_inner_ear', 'm_ax_inf', 'm_ax_sup', 'mid_basel', 'mid_lat', 'mid_prim_inf', 'mid_prim_sup', 'mid_sup', 'optic_chiasm', 'r_caud_head', 'r_corp', 'r_front_pole', 'r_inner_corpus', 'r_lat_ext', 'r_occ_pole', 'r_prim_ext', 'r_sup_ext', 'r_temp_pole', 'right_lateral_inner_ear', 'rostrum', 'rostrum_front', 'top_left', 'top_right']


def getLandmarksFromFCSVFile(file):
    df = pd.read_csv(file, header=None, comment="#")
    if df.shape[1] == 6:
        df.columns = ["label", "x", "y", "z", "sel", "vis"]

    elif df.shape[1] == 14:
        df.columns = [
            "id",
            "x",
            "y",
            "z",
            "ow",
            "ox",
            "oy",
            "oz",
            "vis",
            "sel",
            "lock",
            "label",
            "desc",
            "associatedNodeID",
        ]
    else:
        raise "Please check your input FCSV file"
    df.index = df["label"]
    df = df.reindex(labels=indices)
    # the sign flipping in x and y is to convert RAS(used by slicer) to LPS(used in DICOM and itk)
    xcoord = -1 * df["x"].values.reshape(-1, 1)
    ycoord = -1 * df["y"].values.reshape(-1, 1)
    zcoord = df["z"].values.reshape(-1, 1)
    vec = np.concatenate((xcoord, ycoord, zcoord), axis=1).reshape(-1, 3)
    return vec


###############################################################################


class filesListBrainMRLandmark(object):
    """ A class for managing train files for mri cardio data

        Attributes:
        files_list: Two or on textfiles that contain a list of all images and (landmarks)
        returnLandmarks: Return landmarks if task is train or eval (default: True)
    """

    def __init__(
        self,
        files_list=None,
        returnLandmarks=True,
        agents=2,
        fiducials=None,
        eval=False,
        infDir="../inference",
    ):
        # check if files_list exists
        assert files_list, "There is no directory containing files list"
        self.agents = agents
        # read image filenames
        self.image_files = [line.split("\n")[0] for line in open(files_list[0].name)]
        # read landmark filenames if task is train or eval
        self.returnLandmarks = returnLandmarks
        self.eval = eval
        self.fiducials = fiducials
        if self.returnLandmarks:
            self.landmark_files = [
                line.split("\n")[0] for line in open(files_list[1].name)
            ]
            assert len(self.image_files) == len(
                self.landmark_files
            ), "number of image files is not equal to number of landmark files"
        if infDir[-1] == "/":
            infDir = infDir[:-1]
        if eval:
            if not os.path.isdir(infDir):
                os.mkdir(infDir)
        self.infDir = infDir

    @property
    def num_files(self):
        return len(self.image_files)

    def sample_circular(self, shuffle=False):
        """ return a random sampled ImageRecord from the list of files
        """
        if shuffle:
            indexes = rng.choice(x, len(x), replace=False)
        else:
            indexes = np.arange(self.num_files)

        while True:
            for idx in indexes:
                sitk_image, image = NiftiImage().decode(self.image_files[idx])
                landmarks = []
                images = []
                image_filenames = []
                if self.returnLandmarks:
                    ## transform landmarks to image space if they are in physical space
                    landmark_file = self.landmark_files[idx]
                    if ".fcsv" in landmark_file:
                        all_landmarks = getLandmarksFromFCSVFile(landmark_file)
                    else:
                        all_landmarks = getLandmarksFromTXTFile(landmark_file)
                    # landmark_pc = all_landmarks[14] # landmark index is 13 for ac-point and 14 pc-point
                    # landmark_ac=all_landmarks[13]
                    # transform landmark from physical to image space if required
                    # landmark = sitk_image.TransformPhysicalPointToContinuousIndex(landmark)
                    for i in self.fiducials:
                        landmark = all_landmarks[i]
                        if ".fcsv" in landmark_file:
                            landmark = sitk_image.TransformPhysicalPointToContinuousIndex(
                                landmark
                            )
                        landmarks.append(np.round(landmark).astype("int"))
                    # for i in range(0,self.agents):
                    #     landmark = all_landmarks[i]
                    #     landmarks.append(np.round(landmark).astype('int')) # for 2 agents it will get ac, pc
                else:
                    landmark = None
                # extract filename from path
                # print("agents in datareader", self.agents)
                if self.eval:
                    sitk.WriteImage(
                        sitk_image, self.infDir + "/" + os.path.basename(image.name)
                    )
                for i in range(0, self.agents):
                    images.append(image)
                    image_filenames.append(self.image_files[idx])
                yield images, landmarks, image_filenames, sitk_image.GetSpacing()


class filesListFetalUSLandmark(files):
    """ A class for managing train files for Ozan mri cardio data

        Attributes:
        directory: input data directo
    """

    def __init__(self, files_list=None, returnLandmarks=True, agents=2):

        # assert directory, 'There is no directory containing training files given'
        assert files_list, "There is no directory containing files list"

        self.dir = "/vol/medic01/users/aa16914/projects/tensorpack-medical-gitlab/examples/LandmarkDetection/DQN/data/fetal_brain_us_yuanwei_miccai_2018/"
        self.files_list = [line.split("\n")[0] for line in open(files_list[0])]
        self.images_list = self._listImages()
        self.landmarks_list = self._listLandmarks()
        self.all_landmarks_list = self._listLandmarks_all()
        self.agents = agents

    @property
    def num_files(self):
        return len(self.files_list)

    def _listImages(self):
        # extend directory path
        current_dir = self.dir + "/images"
        image_files = []
        for filename in self.files_list:
            file_path = os.path.join(current_dir, filename + ".nii.gz")
            image_files.append(file_path)

        return image_files

    def _listLandmarks(self):
        # extend directory path
        current_dir = self.dir + "/landmarks"
        landmarks = []
        for filename in self.files_list:
            file_path = os.path.join(current_dir, filename + "_ps.txt")
            points = np.array(extractPointsTXT(file_path))
            # landmark point 12 csp - 11 leftCerebellar - 10 rightCerebellar
            landmark = np.array(points[:, 12])
            landmarks.append(landmark)

        return landmarks

    def _listLandmarks_all(self):
        # extend directory path
        current_dir = self.dir + "/landmarks"
        landmarks = []
        for filename in self.files_list:
            file_path = os.path.join(current_dir, filename + "_ps.txt")
            points = np.array(extractPointsTXT(file_path))
            landmark = np.array(points)  # all landmark point
            landmarks.append(landmark)

        return landmarks

    def sample_circular(self, shuffle=False):
        """ return a random sampled ImageRecord from the list of files
        """
        if shuffle:
            indexes = rng.choice(x, len(x), replace=False)
        else:
            indexes = np.arange(self.num_files)

        while True:
            for idx in indexes:
                sitk_image, image = NiftiImage().decode(self.images_list[idx])
                all_landmarks = self.all_landmarks_list[idx].transpose()
                landmarks = []
                images = []
                image_filenames = []
                for i in range(0, self.agents):
                    landmarks.append(np.round(all_landmarks[i]).astype("int"))
                    images.append(image)
                    image_filenames.append(self.images_list[idx][:-7])
                yield images, landmarks, image_filenames, sitk_image.GetSpacing()

    def sample_circular_all_landmarks(self, shuffle=False):
        """ return a random sampled ImageRecord from the list of files
        """
        if shuffle:
            indexes = rng.choice(x, len(x), replace=False)
        else:
            indexes = np.arange(self.num_files)
        while True:
            for idx in indexes:
                sitk_image, image = niftiImage().decode(self.images_list[idx])
                landmarks = self.all_landmarks_list[idx]
                image_filename = self.images_list[idx][:-7]
                yield image, landmarks, image_filename


##############################################################################


class ImageRecord(object):
    """image object to contain height,width, depth and name """

    pass


class NiftiImage(object):
    """Helper class that provides TensorFlow image coding utilities."""

    def __init__(self):
        pass

    def _is_nifti(self, filename):
        """Determine if a file contains a nifti format image.
        Args
          filename: string, path of the image file
        Returns
          boolean indicating if the image is a nifti
        """
        extensions = [".nii", ".nii.gz", ".img", ".hdr"]
        return any(i in filename for i in extensions)

    def decode(self, filename, label=False):
        """ decode a single nifti image
        Args
          filename: string for input images
          label: True if nifti image is label
        Returns
          image: an image container with attributes; name, data, dims
        """
        image = ImageRecord()
        image.name = filename
        assert self._is_nifti(image.name), "unknown image format for %r" % image.name

        if label:
            sitk_image = sitk.ReadImage(image.name, sitk.sitkInt8)
        else:
            output_size = (300, 300, 300)
            output_spacing = (1, 1, 1)
            offset = np.array(output_size) * np.array(output_spacing) / 2
            identity_transform = sitk.Transform(3, sitk.sitkIdentity)
            sitk_image = sitk.ReadImage(image.name, sitk.sitkFloat32)
            np_image = sitk.GetArrayFromImage(sitk_image)
            itk_image = itk.imread(image.name)
            region = itk_image.GetLargestPossibleRegion()
            index = np.array(region.GetIndex())
            size = np.array(region.GetSize())
            center = index + size / 2
            center = sitk_image.TransformContinuousIndexToPhysicalPoint(center)
            rif = sitk.ResampleImageFilter()
            rif.SetOutputSpacing(output_spacing)
            rif.SetTransform(identity_transform)
            rif.SetOutputOrigin(tuple(np.array(center) - offset))
            rif.SetOutputDirection([1, 0, 0, 0, 1, 0, 0, 0, 1])
            rif.SetSize(output_size)
            sitk_image = rif.Execute(sitk_image)
            # threshold image between p10 and p98 then re-scale [0-255]
            p0 = np_image.min().astype("float")
            p10 = np.percentile(np_image, 10)
            p99 = np.percentile(np_image, 99)
            p100 = np_image.max().astype("float")
            # logger.info('p0 {} , p5 {} , p10 {} , p90 {} , p98 {} , p100 {}'.format(p0,p5,p10,p90,p98,p100))
            sitk_image = sitk.Threshold(
                sitk_image, lower=p10, upper=p100, outsideValue=p10
            )
            sitk_image = sitk.Threshold(
                sitk_image, lower=p0, upper=p99, outsideValue=p99
            )
            sitk_image = sitk.RescaleIntensity(
                sitk_image, outputMinimum=0, outputMaximum=255
            )
        # Convert from [depth, width, height] to [width, height, depth]
        image.data = sitk.GetArrayFromImage(sitk_image).transpose(
            2, 1, 0
        )  # .astype('uint8')
        image.dims = np.shape(image.data)

        return sitk_image, image
