"""
Data manager module for handling data input/output operations.
"""

import os
import pandas as pd
from datetime import datetime


class DataManager:
    OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'output', 'data')

    @staticmethod
    def save_dataframe(df, filename, format='csv'):
        """Save a pandas DataFrame to a file.
        
        Args:
            df: pandas DataFrame to save
            filename: name of the file without extension
            format: file format (default: 'csv')
            
        Returns:
            str: path to the saved file
        """
        # Create output directory if it doesn't exist
        os.makedirs(DataManager.OUTPUT_DIR, exist_ok=True)
        
        # Clean filename and add extension
        # Create timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        clean_filename = f"{filename.replace(' ', '_').lower()}.{format}"
        filepath = os.path.join(DataManager.OUTPUT_DIR, f"{clean_filename}_{timestamp}")
        
        # Save based on format
        if format.lower() == 'csv':
            df.to_csv(filepath, index=True)
        elif format.lower() == 'excel':
            df.to_excel(filepath, index=True)
        else:
            raise ValueError(f"Unsupported format: {format}")
            
        return filepath
