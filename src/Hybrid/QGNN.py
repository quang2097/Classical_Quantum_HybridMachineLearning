from pandas import DataFrame
from pathlib import Path
from src.Hybrid.create_docx import create_docx_QGNN

def QGNN(df:DataFrame, QGNN_output_path:Path, name:str) -> None:
    create_docx_QGNN()