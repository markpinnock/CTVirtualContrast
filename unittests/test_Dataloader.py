import matplotlib.pyplot as plt
import numpy as np
import os
import unittest
import sys; sys.path.append('..')

from utils.DataLoader import PairedLoader, UnpairedLoader


class TestUnpaired(unittest.TestCase):

    def setUp(self) -> None:
        super().setUp()
        self.cv_folds = 3
        self.fold = 2
        self.target = 'AC'
        self.source = 'HQ'
        self.EMPTY_FOLDER = "./FixturesUnpairedEmpty/"
        self.TEST_FOLDER = "./FixturesUnpaired/"
        self.num_subjects = 15
        self.num_subject_imgs = 4

        self.test_config = {
            "DATA_PATH": self.TEST_FOLDER,
            "DATA": {"TARGET": ["AC"], "SOURCE": ["HQ"], "JSON": ""},
            "CV_FOLDS": self.cv_folds,
            "FOLD": self.fold,
            "DOWN_SAMP": 2
            }

        # Set up test data if not already done
        if not os.path.exists(self.TEST_FOLDER + self.target):
            os.makedirs(self.TEST_FOLDER + self.target)
        
        if not os.path.exists(self.EMPTY_FOLDER + self.target):
            os.makedirs(self.EMPTY_FOLDER + self.target)

        if not os.path.exists(self.TEST_FOLDER + self.source):
            os.makedirs(self.TEST_FOLDER + self.source)
        
        if not os.path.exists(self.EMPTY_FOLDER + self.source):
            os.makedirs(self.EMPTY_FOLDER + self.source)
        
        if len(os.listdir(self.TEST_FOLDER)) < self.num_subjects * self.num_subject_imgs:
            for i in range(self.num_subjects):
                for j in range(self.num_subject_imgs):
                    r = np.random.normal(0.0, 5.0, [4, 4, 3])
                    np.save(f"{self.TEST_FOLDER}/{self.target}/T{i:03d}A0{self.target}{j:03d}_000", r)
                    np.save(f"{self.TEST_FOLDER}/{self.source}/T{i:03d}A0{self.source}{j:03d}_000", r)

    def tearDown(self) -> None:
        super().tearDown()
    
    def test_init_neg(self):
        with self.assertRaises(FileNotFoundError):
            dataloader = UnpairedLoader(
                {"DATA_PATH": self.EMPTY_FOLDER,
                "DATA": {"TARGET": ["AC"], "SOURCE": ["HQ"], "JSON": ""},
                "CV_FOLDS": 3,
                "FOLD": 2,
                "DOWN_SAMP": 2
            }, "training")
    
    def test_init(self):
        dataloader = UnpairedLoader(self.test_config, "training")
        self.assertEqual(len(dataloader.unique_ids), self.num_subjects)
        self.assertEqual(len(dataloader.data["targets"]), self.num_subjects * self.num_subject_imgs)

    def test_train_val_split(self):
        train_dataloader = UnpairedLoader(self.test_config, "training")
        train_dataloader.train_val_split(None)
        train_slice = slice(0, (self.num_subjects * self.num_subject_imgs * self.fold) // self.cv_folds)
        train_data = train_dataloader.data["targets"][train_slice]
        self.assertEqual(train_dataloader.fold_data["targets"], train_data)

        val_dataloader = UnpairedLoader(self.test_config, "validation")
        val_dataloader.train_val_split(None)
        val_slice = slice((self.num_subjects * self.num_subject_imgs * self.fold) // self.cv_folds, self.num_subjects * self.num_subject_imgs)
        val_data = val_dataloader.data["targets"][val_slice]
        self.assertEqual(val_dataloader.fold_data["targets"], val_data)

        single_fold_dataloader = UnpairedLoader(
            {"DATA_PATH": self.TEST_FOLDER,
            "DATA": {"TARGET": ["AC"], "SOURCE": ["HQ"], "JSON": ""},
            "CV_FOLDS": 1,
            "FOLD": 0,
            "DOWN_SAMP": 2
        }, "training")

        single_fold_dataloader.train_val_split(None)
        self.assertEqual(single_fold_dataloader.data["targets"], single_fold_dataloader.fold_data["targets"])
  
    def test_train_val_split_neg(self):

        with self.assertRaises(ValueError):
            dataloader = UnpairedLoader(
                {"DATA_PATH": self.TEST_FOLDER,
                "DATA": {"TARGET": ["AC"], "SOURCE": ["HQ"], "JSON": ""},
                "CV_FOLDS": 1,
                "FOLD": 1,
                "DOWN_SAMP": 2
            }, "training")

            dataloader.train_val_split()

        with self.assertRaises(ValueError):
            dataloader = UnpairedLoader(
                {"DATA_PATH": self.TEST_FOLDER,
                "DATA": {"TARGET": ["AC"], "SOURCE": ["HQ"], "JSON": ""},
                "CV_FOLDS": 0,
                "FOLD": -1,
                "DOWN_SAMP": 2
            }, "training")

            dataloader.train_val_split()

        with self.assertRaises(ValueError):
            dataloader = UnpairedLoader(
                {"DATA_PATH": self.TEST_FOLDER,
                "DATA": {"TARGET": ["AC"], "SOURCE": ["HQ"], "JSON": ""},
                "CV_FOLDS": 3,
                "FOLD": 2,
                "DOWN_SAMP": 2
            }, "fail")

            dataloader.train_val_split()

    def test_img_pairer(self):
        dataloader = UnpairedLoader(self.test_config, "training")
        source = "T000A0HQ000_000.npy"
        imgs = dataloader.img_pairer(source)
        self.assertEqual(imgs["target"][0:6], imgs["source"][0:6])
        self.assertEqual(imgs["target"][-7:], imgs["source"][-7:])


#-------------------------------------------------------------------------

class Testpaired(unittest.TestCase):

    def setUp(self) -> None:
        super().setUp()
        self.cv_folds = 3
        self.fold = 2
        self.target = 'AC'
        self.source = 'HQ'
        self.TEST_FOLDER = "./FixturesUnpaired/"
        self.num_subjects = 15
        self.num_subject_imgs = 4

        self.test_config = {
            "DATA_PATH": self.TEST_FOLDER,
            "DATA": {"TARGET": ["AC"], "SOURCE": ["HQ"], "JSON": ""},
            "CV_FOLDS": self.cv_folds,
            "FOLD": self.fold,
            "DOWN_SAMP": 2
            }

        if not os.path.exists(self.TEST_FOLDER + self.target):
            os.makedirs(self.TEST_FOLDER + self.target)

        if not os.path.exists(self.TEST_FOLDER + self.source):
            os.makedirs(self.TEST_FOLDER + self.source)
        
        if len(os.listdir(self.TEST_FOLDER)) < self.num_subjects * self.num_subject_imgs:
            for i in range(self.num_subjects):
                for j in range(self.num_subject_imgs):
                    r = np.random.normal(0.0, 5.0, [4, 4, 3])
                    np.save(f"{self.TEST_FOLDER}/{self.target}/T{i:03d}A0{self.target}{j:03d}_000", r)
                    np.save(f"{self.TEST_FOLDER}/{self.source}/T{i:03d}A0{self.source}{j:03d}_000", r)

    def tearDown(self) -> None:
        super().tearDown()
    
    def test_img_pairer(self):
        dataloader = PairedLoader(self.test_config, "training")
        source = "T000A0HQ000_000.npy"
        imgs = dataloader.img_pairer(source)
        self.assertEqual(imgs["target"][0:6], imgs["source"][0:6])
        self.assertNotEqual(imgs["target"][6:8], imgs["source"][6:8])
    
    # TODO: test generator/normalisation


if __name__ == "__main__":

    """ Quick routine to visually check output of generator """

    test_config = {
        "DATA_PATH": "D:/ProjectImages/SyntheticContrast",
        "DATA": {"TARGET": ["AC"], "SOURCE": ["HQ"], "JSON": ""},
        "CV_FOLDS": 3,
        "FOLD": 2,
        "DOWN_SAMP": 2
        }

    d = PairedLoader(test_config, "training")
    d.set_normalisation(norm_type="std")

    for imgs in d.data_generator():
        plt.subplot(1, 2, 1)
        plt.imshow(imgs[0][:, :, 5], cmap="gray")
        plt.subplot(1, 2, 2)
        plt.imshow(imgs[1][:, :, 5], cmap="gray")
        plt.show()