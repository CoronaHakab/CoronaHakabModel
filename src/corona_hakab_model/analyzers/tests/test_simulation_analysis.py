from unittest import TestCase
import os
from corona_hakab_model.analyzers.multi_simulation_analysis import create_comparison_files, plot_minmax_barchart_single_param
from project_structure import OUTPUT_FOLDER


class TestSimAnalysis(TestCase):
    def test_create_comparison_files(self):

        def count_dirs_in_path(dir_path) -> int:
            count = 0
            for path in os.listdir(dir_path):
                if os.path.isdir(os.path.join(dir_path, path)):
                    count += 1
            return count

        n_folders_in_output_before = count_dirs_in_path(OUTPUT_FOLDER)
        create_comparison_files()
        n_folders_in_output_after = count_dirs_in_path(OUTPUT_FOLDER)
        self.assertEqual(n_folders_in_output_before + 1, n_folders_in_output_after)

