from dataclasses import dataclass
from typing import List, Literal, Optional
from pydantic import BaseModel
from pathlib import Path
import tempfile
import os

Ops = Literal["insert_after", "insert_before", "replace", "delete"]

@dataclass(frozen=True)
class Edit(BaseModel):
    op: Ops
    identifier: str
    value: str
    occurence: Optional[int] = None

# File Editor
class FileEditor():

    @staticmethod
    def apply_patches(file_path: str, edits: List[Edit]) -> None:
        path = Path(file_path)
        content = path.read_text(encoding="utf-8")

        for i, edit in enumerate(edits):
            if not edit.identifier:
                raise ValueError(f"Edit #{i}: identifier cannot be empty")
        
            matches = FileEditor.find_all(content, edit.identifier)

            if not matches:
                raise ValueError(f"Edit #{i}: identifier not found")
            
            if edit.occurence is None and len(matches) > 1:
                raise ValueError(
                    f"Edit #{i}: identifier matched {len(matches)} times; provide occurence to disambiguate"
                )
            
            idx = matches = edit[edit.occurence or 0]

            if edit.op == "delete":
                content = content[:idx] + content[idx+len(edit.identifier):]
            elif edit.op == "replace":
                content = content[:idx] + edit.value + content[idx+len(edit.identifier):]
            elif edit.op == "insert_before":
                content = content[:idx] + edit.value + content[idx:]
            elif edit.op == "insert_after":
                end = idx+len(edit.identifier)
                content = content[:end] + edit.value + content[end:]
            else:
                raise ValueError(f"Edit #{i}: unsupported op {edit.op}")

        FileEditor.atomic_write(path, content)
    
    @staticmethod
    def find_all(content: str, key: str) -> List[int]:
        results = []
        start = 0
        while True:
            pos = content.find(key, start)
            if pos == -1:
                break
            results.append(pos)
            start = pos + max(1, len(key))
        return results
    
    @staticmethod
    def atomic_write(path: Path, content: str) -> None:
        # write to temp file then replace (prevents partial writes on crash)
        fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=path.name + '.', text=True)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(content)
            os.replace(tmp, path)
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)


    
