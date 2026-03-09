import os
import sys

def configure_path():
    # Always use the 64_lib directory inside the dlls folder relative to this script
    relative_path_to_dlls = '.' + os.sep + 'dlls' + os.sep + '64_lib'

    absolute_path_to_file_directory = os.path.dirname(os.path.abspath(__file__))
    absolute_path_to_dlls = os.path.abspath(os.path.join(absolute_path_to_file_directory, relative_path_to_dlls))

    os.environ['PATH'] = absolute_path_to_dlls + os.pathsep + os.environ.get('PATH', '')

    try:
        # Python 3.8+ method to add DLL directory explicitly
        os.add_dll_directory(absolute_path_to_dlls)
    except AttributeError:
        # Older Python versions do not have this method, ignore
        pass
