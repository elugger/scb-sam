"""Real filesystem regression for both phases; never touches remote targets."""
import tempfile
import unittest
from pathlib import Path, PurePosixPath
from unittest.mock import patch

import launch_hpq as launch


class PathOperationsTests(unittest.TestCase):
    def test_all_production_targets_are_concrete(self):
        for phase in ('bridge', 'formal'):
            for name, target in launch.paths(phase).items():
                with self.subTest(phase=phase, target=name):
                    self.assertIsInstance(target, Path)
                    self.assertTrue(callable(target.mkdir))

    def test_real_directory_creation_and_residual_rejection(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            # Preserve the production contract's PurePosixPath input type.
            with patch.object(launch, 'BASE', PurePosixPath(root.as_posix()) / 'output'), \
                 patch.object(launch, 'STORE', PurePosixPath(root.as_posix()) / 'formal'), \
                 patch.object(launch, 'BRIDGE_STORE', PurePosixPath(root.as_posix()) / 'bridge'), \
                 patch.object(launch, 'CONTROL', root / 'control'):
                for phase in ('bridge', 'formal'):
                    targets = launch.paths(phase)
                    for name in ('output', 'store'):
                        with self.subTest(phase=phase, target=name):
                            targets[name].mkdir(parents=True, exist_ok=False)
                            self.assertTrue(targets[name].is_dir())
                            with self.assertRaises(FileExistsError):
                                targets[name].mkdir(parents=True, exist_ok=False)


if __name__ == '__main__':
    unittest.main()
