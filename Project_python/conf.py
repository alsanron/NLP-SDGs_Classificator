import sys, os

def import_paths():
    # Imports the required paths by all the project-modules in a relative mode.
    # @warning This function should not be modified by the user
    sys.path.append(os.path.realpath('codes'))
    sys.path.append(os.path.realpath('analysis'))
    sys.path.append(os.path.realpath('datasets'))
    sys.path.append(os.path.realpath('models'))
    sys.path.append(os.path.realpath('ref'))
    sys.path.append(os.path.realpath('test'))

def get_paths():
    # Returns the paths
    paths = dict()
    paths["ref"] = "ref/"
    paths["Abstracts"] = paths["ref"] + "Abstracts/"
    paths["Nature"] = paths["ref"] + "Nature/"
    paths["SDGs_inf"] = paths["ref"] + "SDGs_information/"
    paths["out"] = "out/"
    paths["model"] = "models/"
    paths["manual"] = paths["ref"] + "Manual_selected/"
    paths["test"] = "test/"
    return paths

