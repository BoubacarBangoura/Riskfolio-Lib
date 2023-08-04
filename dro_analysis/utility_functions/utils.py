import pickle
import os
import shutil
from collections import Counter
import random

from dro_analysis import paths

import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt


def get_super(x):
    """returns a super character that can be printed"""
    normal = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+-=()"
    super_s = "ᴬᴮᶜᴰᴱᶠᴳᴴᴵᴶᴷᴸᴹᴺᴼᴾQᴿˢᵀᵁⱽᵂˣʸᶻᵃᵇᶜᵈᵉᶠᵍʰᶦʲᵏˡᵐⁿᵒᵖ۹ʳˢᵗᵘᵛʷˣʸᶻ⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾"
    res = x.maketrans(''.join(normal), ''.join(super_s))
    return x.translate(res)


def get_sub(x):
    """returns a sub character that can be printed"""
    normal = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+-=()"
    sub_s = "ₐ₈CDₑբGₕᵢⱼₖₗₘₙₒₚQᵣₛₜᵤᵥwₓᵧZₐ♭꜀ᑯₑբ₉ₕᵢⱼₖₗₘₙₒₚ૧ᵣₛₜᵤᵥwₓᵧ₂₀₁₂₃₄₅₆₇₈₉₊₋₌₍₎"
    res = x.maketrans(''.join(normal), ''.join(sub_s))
    return x.translate(res)


def load_pickle(where):
    file = open(where, 'rb')
    what = pickle.load(file)
    file.close()
    return what


def save_backup(file_path: str):
    backup_file_path = file_path.replace('data', 'data_backup', 1)
    try:
        shutil.copy(file_path, backup_file_path)
        print("Successfully backed up")
    except shutil.SameFileError:
        print("BACKUP ISSUE: Source and destination represents the same file.")
    except PermissionError:
        print("BACKUP ISSUE: Permission denied.")
    except:
        print("BACKUP ISSUE: Error occurred while copying file.")


def save_pickle(what, where):
    if os.path.exists(where):
        backup = None
        while backup not in ['y', 'n']:
            backup = input('Do you want to save a backup of the previous version? [y/n]:')
        if backup == 'y':
            save_backup(where)

    file = open(where, 'wb')
    pickle.dump(what, file)
    file.close()


def load_market(market_name: str, folder_path=paths.MARKETS):
    return load_pickle(os.path.join(folder_path, market_name))


def load_portfolio(portfolio_name: str, folder_path=paths.PORTFOLIOS):
    return load_pickle(os.path.join(folder_path, portfolio_name))
