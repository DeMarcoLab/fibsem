from PyQt5.QtCore import Qt, QAbstractTableModel, pyqtSignal
import pandas as pd
import numpy as np

class PandasTableModel(QAbstractTableModel):
    """Custom table model to sync PyQt5 TableView with pandas DataFrame"""
    dataChanged = pyqtSignal(pd.DataFrame)
    
    def __init__(self, df=pd.DataFrame(), display_columns=None):
        super().__init__()
        self._df = df.copy()
        self._df.index = range(1, len(df) + 1)  # 1-based index
        self.row_order = list(range(len(df)))
        
        # Set up display columns
        self.display_columns = display_columns
        if self.display_columns is None:
            self.display_columns = list(df.columns)
        else:
            # Verify all columns exist
            missing_cols = set(self.display_columns) - set(df.columns)
            if missing_cols:
                raise ValueError(f"Columns not found in DataFrame: {missing_cols}")
            
        # Create a view of the DataFrame with only the display columns
        self._display_df = self._df[self.display_columns]

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid():
            return None
            
        if role == Qt.DisplayRole:
            # Map the view index to the actual DataFrame index
            row = self.row_order[index.row()]
            value = self._display_df.iloc[row, index.column()]
            
            if pd.isna(value):
                return ''
            if isinstance(value, (float, np.floating)):
                return f"{value:.2f}"
            return str(value)
            
        return None

    def setData(self, index, value, role=Qt.EditRole):
        if role == Qt.EditRole:
            try:
                row = self.row_order[index.row()]
                col_name = self.display_columns[index.column()]
                current_dtype = self._df[col_name].dtype
                
                if pd.api.types.is_numeric_dtype(current_dtype):
                    value = pd.to_numeric(value)
                
                # Update both DataFrames
                self._df.iloc[row, self._df.columns.get_loc(col_name)] = value
                self._display_df.iloc[row, index.column()] = value
                
                self.dataChanged.emit(self._df)
                return True
            except (ValueError, TypeError):
                return False
        return False

    def rowCount(self, parent=None):
        return len(self._df)

    def columnCount(self, parent=None):
        return len(self.display_columns)

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if role == Qt.DisplayRole:
            if orientation == Qt.Horizontal:
                return str(self.display_columns[section])
            else:
                return str(section + 1)
        elif role == Qt.TextAlignmentRole and orientation == Qt.Vertical:
            return Qt.AlignRight | Qt.AlignVCenter
        return None

    def flags(self, index):
        return Qt.ItemIsEditable | Qt.ItemIsEnabled | Qt.ItemIsSelectable | \
               Qt.ItemIsDragEnabled | Qt.ItemIsDropEnabled

    def update_df(self, new_df):
        """Update the model's DataFrame and refresh the view"""
        self.layoutAboutToBeChanged.emit()
        self._df = new_df.copy()
        self._df.index = range(1, len(new_df) + 1)
        self._display_df = self._df[self.display_columns]
        self.row_order = list(range(len(new_df)))
        self.layoutChanged.emit()

    def moveRow(self, source_row, dest_row):
        """Handle row reordering"""
        if source_row == dest_row:
            return

        self.layoutAboutToBeChanged.emit()
        
        # Update the row order
        moving_row = self.row_order.pop(source_row)
        self.row_order.insert(dest_row, moving_row)
        
        # Create new DataFrame with reordered rows
        temp_df = self._df.iloc[self.row_order].copy()
        
        # Reset the index to be 1-based consecutive numbers
        temp_df.index = range(1, len(temp_df) + 1)
        
        # Update both DataFrames and row order
        self._df = temp_df
        self._display_df = self._df[self.display_columns]
        self.row_order = list(range(len(self._df)))
        
        self.layoutChanged.emit()
        self.dataChanged.emit(self._df)

    def get_data(self):
        """Return the current state of the full DataFrame"""
        return self._df.copy()

    def set_display_columns(self, columns):
        """Update which columns are displayed in the view"""
        # Verify all columns exist
        missing_cols = set(columns) - set(self._df.columns)
        if missing_cols:
            raise ValueError(f"Columns not found in DataFrame: {missing_cols}")
            
        self.layoutAboutToBeChanged.emit()
        self.display_columns = list(columns)
        self._display_df = self._df[self.display_columns]
        self.layoutChanged.emit()
