import os
import sys

def configure_path():
    # Updated to reference the thorlabs_lib/camera/dlls/64_lib directory
    # os.path.join handles the slashes correctly for Windows/Linux
    relative_path_to_dlls = os.path.join('thorlabs_lib', 'camera', 'dlls', '64_lib')

    # Get the directory where this script is located
    absolute_path_to_file_directory = os.path.dirname(os.path.abspath(__file__))
    
    # Create the absolute path to the Thorlabs DLLs
    absolute_path_to_dlls = os.path.abspath(os.path.join(absolute_path_to_file_directory, relative_path_to_dlls))

    # Verify the path exists to avoid silent failures during hardware initialization
    if not os.path.exists(absolute_path_to_dlls):
        print(f"Warning: DLL directory not found at {absolute_path_to_dlls}")

    # Add to environment PATH for general library discovery
    os.environ['PATH'] = absolute_path_to_dlls + os.pathsep + os.environ.get('PATH', '')

    try:
        # Crucial for Python 3.8+ on Windows to load hardware drivers
        os.add_dll_directory(absolute_path_to_dlls)
    except AttributeError:
        # Older Python versions handle DLLs via the PATH environment variable
        pass

    return absolute_path_to_dlls