from pandas import DataFrame
from pathlib import Path
from src.Classical.create_docx import create_docx_GNN

def GNN(df:DataFrame, output_path:Path, name:str) -> None:
    create_docx_GNN()