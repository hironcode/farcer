import os
from pathlib import Path


class PathHelper:
    def __init__(self) -> None:
        self.DEPTH = 2
        self.rootdir = self.get_root_dir()
    
    def get_root_dir(self) -> Path:
        # path to this file

        path = os.path.realpath(__file__)
        pathlist = path.rsplit("/", self.DEPTH)
        rootdir = Path(pathlist[0])
        return rootdir
    
    def get_target_dir(self, target_path:str) -> Path:
        """
        target_path = Path from the root directory (exclusive)
        """
        path  = Path(os.path.join(self.rootdir, target_path))
        return path
    
    def remove_unnecessary_paths(self, path) -> Path:
        """
        Removes unnecessary paths from the given path and returns the path from the root directory.
        """
        return os.path.relpath(path, self.rootdir)