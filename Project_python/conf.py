
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
