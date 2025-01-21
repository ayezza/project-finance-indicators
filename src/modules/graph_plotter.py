"""
Graph plotter module providing plotting utilities.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime


class GraphPlotter:
    OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'output', 'plots')

    @staticmethod
    def plot_2d_graph(x_data=None, y_data=None, plot_params=None):
        if x_data is None or y_data is None:
            return None, None
        if plot_params is None:
            plot_params = {}

        default_plot_params = {
            'title': '', 
            'fontsize': '12', 
            'fontname': 'arial', 
            'color': '#000000',
            'x_label': 'variable',
            'y_label': 'Value', 
            'style': '+-b', 
            'x_step': (max(x_data)-min(x_data))/10
        }

        for key in default_plot_params:
            if key in plot_params and plot_params[key] is not None:
                default_plot_params[key] = plot_params[key]

        fig, ax = plt.subplots(1, 1, sharex=False, sharey=False,
                              subplot_kw={'facecolor': 'white'})
        plt.grid(True, which='major', axis='both', lw=1, ls='--', c='.75')
        ax.plot(x_data, y_data, default_plot_params['style'])
        
        x_ticks = np.arange(x_data[0], x_data[-1] + default_plot_params['x_step'], 
                           default_plot_params['x_step'])
        ax.set_xticks(x_ticks)
        
        ax.set_xlabel(default_plot_params['x_label'], labelpad=5, fontsize=14, 
                     fontname='serif', color="blue")
        ax.set_ylabel(default_plot_params['y_label'], labelpad=5, fontsize=14, 
                     fontname='serif', color="red")
        ax.set_title(default_plot_params['title'], 
                    fontsize=default_plot_params['fontsize'],
                    fontname=default_plot_params['fontname'], 
                    color=default_plot_params['color'])

        return fig, ax

    @staticmethod
    def add_annotation_to_graph(fig=None, ax=None, p=None, text='', xytext=None,
                              arrowprops=None):
        if ax is None or p is None:
            return
        
        if xytext is None:
            xytext = (p[0]-20, p[1]-20)
        if arrowprops is None:
            arrowprops = dict(arrowstyle="->", connectionstyle="arc3, rad=.5")

        if p[1] is not None:
            ax.axhline(p[1], ls='-.')
        if p[0] is not None:
            ax.axvline(p[0], ls='-.')
            
        ax.annotate(text, fontsize=12, family="serif", xy=p, xycoords="data", 
                   textcoords="offset points", xytext=xytext, arrowprops=arrowprops)

    @staticmethod
    def finalize_and_save(fig, filename, dpi=300, format='png'):
        """Save the figure to a file in the output directory.
        
        Args:
            fig: matplotlib figure object
            filename: name of the file without extension
            dpi: dots per inch for the saved figure
            format: file format (e.g., 'png', 'pdf', 'svg')
        """
        # Create output directory if it doesn't exist
        os.makedirs(GraphPlotter.OUTPUT_DIR, exist_ok=True)
        
        # Clean filename and add timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        clean_filename = f"{filename.replace(' ', '_').lower()}_{timestamp}.{format}"
        filepath = os.path.join(GraphPlotter.OUTPUT_DIR, clean_filename)
        
        # Adjust layout to prevent text cutoff
        plt.tight_layout()
        
        # Save the figure
        fig.savefig(filepath, dpi=dpi, bbox_inches='tight', format=format)
        print(f"Plot saved to: {filepath}")
        
        # Close the figure to free memory
        plt.close(fig)
