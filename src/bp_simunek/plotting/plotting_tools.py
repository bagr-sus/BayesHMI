import os
import traceback
import logging
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from ..definitions import ROOT_DIR

def graphs_path() -> None:
    return os.path.join(ROOT_DIR, "data", "graphs")

def save_plot(
        filename: str,
        fig: plt.Figure = None,
        folder_path: str = graphs_path()) -> None:

    # if path doesn't exist, create it
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    try:
        if fig is None:
            plt.savefig(os.path.join(folder_path, filename), dpi=300)
        else:
            fig.savefig(os.path.join(folder_path, filename), dpi=300)
        logging.info("Succesfully saved plot %s at %s.", filename, folder_path)
    except:
        logging.error("Failed to save plot at %s at %s!", filename, folder_path)
        logging.error(traceback.format_exc())

def save_plots_pdf_pages(
        filename: str,
        figs: list,
        folder_path: str = graphs_path()) -> None:

    if not figs:
        return

    # if path doesn't exist, create it
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    path = os.path.join(folder_path, filename)
    try:
        with PdfPages(path) as pdf:
            for fig in figs:
                pdf.savefig(fig)
                plt.close(fig)
        logging.info("Succesfully saved plot %s at %s.", filename, folder_path)
    except:
        logging.error("Failed to save plot at %s at %s!", filename, folder_path)
        logging.error(traceback.format_exc())
