import sys
import subprocess
import importlib

class DependencyInstaller:
    """
    Handles installation and import of required Python packages.
    """

    def __init__(self, required_packages: dict):
        """
        Initializes the installer with a dictionary of packages.

        Args:
            required_packages (dict): Mapping of pip package names to import names.
        """
        self.required_packages = required_packages

    def install_and_import(self, package, import_name=None):
        """
        Installs and imports a package.

        Args:
            package (str): Package name for pip.
            import_name (str): Module name for import (optional).
        """
        import_name = import_name or package
        try:
            importlib.import_module(import_name)
        except ImportError:
            print(f"Installing missing package: {package}")
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", "--quiet", "--no-warn-script-location", package
            ])
            importlib.import_module(import_name)

    def install_all(self):
        """
        Installs and imports all required packages.
        """
        for pkg, module in self.required_packages.items():
            self.install_and_import(pkg, module)
