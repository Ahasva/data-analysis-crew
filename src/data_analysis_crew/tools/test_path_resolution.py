from pathlib import Path

project_root = Path(__file__).resolve().parents[3]

FILE_NAME = "diabetes.csv"
FOLDER_NAME = "knowledge"

csv_path = project_root / 'knowledge' / FILE_NAME

print("📂 Project root:", project_root)
print("📄 Resolved CSV path:", csv_path.resolve())
print("✅ File exists:", csv_path.exists())
