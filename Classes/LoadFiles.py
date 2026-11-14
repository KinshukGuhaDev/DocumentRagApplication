from unittest import case
import csv
import json
import re
from typing import List

from langchain_community.document_loaders import (
    CSVLoader,
    Docx2txtLoader,
    JSONLoader,
    PyMuPDFLoader,
    TextLoader,
    UnstructuredExcelLoader,
)
from pathlib import Path
import hashlib
from config import DIRECTORY_PATH, FILE_TYPE, IS_SINGLE_FILE, SINGLE_FILE_PATH
from langchain_core.documents import Document
import openpyxl

class LoadFiles:
    def __init__(
        self, 
        directory_path: str = DIRECTORY_PATH, 
        file_type: str = FILE_TYPE, 
        is_single_file: bool = IS_SINGLE_FILE, 
        single_file_path: str = SINGLE_FILE_PATH
    ):
        self.directory_path = directory_path
        self.file_type = file_type
        self.is_single_file = is_single_file
        self.single_file_path = single_file_path
        
    def _validate_file(self, file: Path) -> bool:
        supported_types = {
            "pdf": {".pdf"},
            "docx": {".docx"},
            "txt": {".txt"},
            "csv": {".csv"},
            "json": {".json"},
            "xlsx": {".xlsx"},
            "xls": {".xls"},
            "excel": {".xlsx", ".xls"},
            "sql": {".sql"},
            "all": {".pdf", ".docx", ".txt", ".csv", ".json", ".xlsx", ".xls", ".sql"},
        }

        file_type = self.file_type.lower()
        file_name = file.name

        if not file.is_file():
            print(f"File '{file_name}' does not exist or is not a regular file.")
            return False

        allowed_extensions = supported_types.get(file_type)
        if allowed_extensions is None:
            print(f"Unsupported file type: {file_type}")
            return False

        if file.suffix.lower() not in allowed_extensions:
            print(
                f"Invalid file '{file_name}'. "
                f"Expected: {', '.join(sorted(allowed_extensions))}"
            )
            return False
        return True

    def _get_loader(self, file_type):
        match file_type.strip().lower().lstrip("."):
            case "pdf":
                return PyMuPDFLoader
            case "docx":
                return Docx2txtLoader
            case "txt" | "sql":
                return TextLoader
            case "csv":
                return CSVLoader
            case "json":
                return JSONLoader
            case "xlsx" | "xls":
                return UnstructuredExcelLoader
            case _:
                return None

    def _load_csv_file(self, file: Path) -> List[Document]:
        documents = []
        with file.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames:
                raise ValueError(f"CSV file has no header row: {file.name}")

            for row_number, row in enumerate(reader, start=2):
                fields = [
                    f"{name}: {row.get(name, '')}"
                    for name in reader.fieldnames
                ]
                text = "\n".join(fields).strip()
                if text:
                    documents.append(Document(
                        page_content=text,
                        metadata={
                            "row_number": row_number,
                            "columns": ", ".join(reader.fieldnames),
                            "content_type": "csv_record",
                        },
                    ))
        return documents

    def _load_json_file(self, file: Path) -> List[Document]:
        with file.open("r", encoding="utf-8-sig") as handle:
            value = json.load(handle)

        documents = []

        def visit(item, path):
            if isinstance(item, dict):
                text = "\n".join(
                    f"{key}: {json.dumps(val, ensure_ascii=False) if isinstance(val, (dict, list)) else val}"
                    for key, val in item.items()
                ).strip()
                if text:
                    documents.append(Document(
                        page_content=text,
                        metadata={
                            "json_path": path or "$",
                            "content_type": "json_object",
                        },
                    ))
                for key, val in item.items():
                    if isinstance(val, (dict, list)):
                        visit(val, f"{path}.{key}" if path else f"$.{key}")
            elif isinstance(item, list):
                for index, val in enumerate(item):
                    visit(val, f"{path}[{index}]")
            else:
                documents.append(Document(
                    page_content=str(item),
                    metadata={
                        "json_path": path or "$",
                        "content_type": "json_value",
                    },
                ))

        visit(value, "")
        return documents

    def _load_excel_file(self, file: Path) -> List[Document]:
        workbook = openpyxl.load_workbook(
            filename=file,
            read_only=True,
            data_only=False,
        )
        documents = []

        for worksheet in workbook.worksheets:
            rows = list(worksheet.iter_rows(values_only=True))
            rows = [row for row in rows if any(value is not None for value in row)]
            if not rows:
                continue

            headers = [
                str(value).strip() if value is not None else f"column_{i + 1}"
                for i, value in enumerate(rows[0])
            ]

            for row_number, row in enumerate(rows[1:], start=2):
                fields = []
                for index, header in enumerate(headers):
                    value = row[index] if index < len(row) else None
                    fields.append(f"{header}: {value if value is not None else ''}")

                text = "\n".join(fields).strip()
                if text:
                    documents.append(Document(
                        page_content=text,
                        metadata={
                            "worksheet": worksheet.title,
                            "row_number": row_number,
                            "columns": ", ".join(headers),
                            "content_type": "spreadsheet_record",
                        },
                    ))

        workbook.close()
        return documents

    def _load_sql_file(self, file: Path) -> List[Document]:
        text = file.read_text(encoding="utf-8-sig")
        statements = [part.strip() for part in re.split(r"(?<=;)\s*", text) if part.strip()]
        return [
            Document(
                page_content=statement,
                metadata={
                    "statement_number": index,
                    "content_type": "sql_statement",
                },
            )
            for index, statement in enumerate(statements, start=1)
        ]


    def load_files(self):
        print("_"*50)
        print("")
        print("Loading files...")
        if self.is_single_file:
            file = Path(self.single_file_path)
            files = (file,)
        else:
            directory = Path(self.directory_path)
            if not directory.is_dir():
                print(f"Directory not found: {directory}")
                return False
            # Iterate through files without building a list of every path.
            files = directory.rglob("*")

        all_documents = []
        processed_files = 0

        print("-"*50)

        for i, file in enumerate(files):
            if not file.is_file():
                continue

            file_type = file.suffix.lower().lstrip(".")
            loader_class = self._get_loader(file_type)

            if loader_class is None:
                if self.is_single_file:
                    print(f"❌ Unsupported file type: {file.suffix}")
                    return False
                continue

            validation_type = file_type if self.file_type.lower() == "all" else self.file_type
            original_type = self.file_type
            self.file_type = validation_type
            is_valid = self._validate_file(file)
            self.file_type = original_type

            if not is_valid:
                if self.is_single_file:
                    return False
                continue

            if i != 0:
                print("\n")
            print(f"🔃 Processing file: {file.name} from: {file.parent}")

            try:
                loader_options = {}
                if file_type == "csv":
                    documents = self._load_csv_file(file)
                elif file_type == "json":
                    documents = self._load_json_file(file)
                elif file_type in ("xlsx", "xls"):
                    documents = self._load_excel_file(file)
                elif file_type == "sql":
                    documents = self._load_sql_file(file)
                else:
                    if file_type in ("txt",):
                        loader_options = {
                            "encoding": "utf-8",
                        }

                    loader = loader_class(str(file), **loader_options)
                    documents = loader.load()

                if file_type in ("txt",):
                    loader_options = {
                        "encoding": "utf-8",
                    }

                for doc in documents:
                    doc.metadata["source_file"] = file.name
                    doc.metadata["source_path"] = str(file.resolve())
                    doc.metadata["file_type"] = file_type
                    doc.metadata["document_id"] = hashlib.sha256(
                        str(file.resolve()).encode("utf-8")
                    ).hexdigest()

                all_documents.extend(documents)
                processed_files += 1

                print(f"✅ Loaded {len(documents)} documents from {file.name}.")
            except Exception as e:
                print(f"❌ Error processing {file}: {e}")
                return False

        print("-"*50)
        print(
            f"✅ Loaded {len(all_documents)} documents "
            f"from {processed_files} files."
        )
        print("_"*50)
        print("\n")
        return all_documents
