import unittest
from unittest import mock
from unittest.mock import patch

import numpy as np
from PIL import Image

from pyaugment.modules.region_proposer.seem_region_proposer import SEEMRegionProposer


class TestSEEMRegionProposer(unittest.TestCase):
    def setUp(self):
        self.seem_proposer = SEEMRegionProposer("mock_config.yaml")
        self.seem_proposer.object_categories = {"test_class": 0}

    @patch("PIL.Image.open")
    @patch(
        "pyaugment.modules.region_proposer.seem_region_proposer.SEEMRegionProposer._infere_mask"
    )
    def test_get_detections(self, mock_infere, mock_open):
        mock_open.return_value = Image.new("RGB", (512, 512))
        mock_infere.return_value = np.zeros(shape=(512, 512))

        detections = self.seem_proposer._get_detections("", ["test_class."])
        self.assertIsNone(detections)

    @patch(
        "pyaugment.modules.region_proposer.seem_region_proposer.SEEMRegionProposer._get_detections"
    )
    def test_propose_region(self, mock_get_detections):
        mock_get_detections.return_value = None
        test_images = ["test_image_1", "test_image_2"]
        with mock.patch("pathlib.Path.glob", return_value=test_images):
            annotated_images = self.seem_proposer.propose_region(
                "test_path", ["test_class."]
            )
            self.assertEqual(mock_get_detections.call_count, len(test_images))
            self.assertFalse(annotated_images)


if __name__ == "__main__":
    unittest.main()
