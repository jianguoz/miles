import json
import argparse


def convert_data_format(input_path: str, output_path: str):
    """Convert JSON dict format to JSONL format for Miles SFT training.

    Input format (JSON dict):
        {
            "unique_id_1": {"messages": [...], "tools": [...], ...},
            "unique_id_2": {"messages": [...], "tools": [...], ...},
            ...
        }

    Output format (JSONL):
        {"id": "unique_id_1", "messages": [...], "tools": [...], ...}
        {"id": "unique_id_2", "messages": [...], "tools": [...], ...}
        ...

    All original keys are preserved.
    """
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    with open(output_path, "w", encoding="utf-8") as f:
        for unique_id, sample_data in data.items():
            output_sample = {"id": unique_id}
            for key, value in sample_data.items():
                if key == "id":
                    if value != unique_id:
                        raise ValueError(f"id {value} is not equal to {unique_id} for sample {unique_id}")
                else:
                    output_sample[key] = value
            f.write(json.dumps(output_sample, ensure_ascii=False) + "\n")

    print(f"Converted {len(data)} samples from {input_path} to {output_path}")


if __name__ == "__main__":
    """
    python convert_data_format.py --input gorilla_multi.json --output gorilla_multi.jsonl
    """
    parser = argparse.ArgumentParser(description="Convert JSON to JSONL for Miles SFT")
    parser.add_argument("--input", type=str, required=True, help="Input JSON file path")
    parser.add_argument("--output", type=str, required=True, help="Output JSONL file path")
    args = parser.parse_args()

    convert_data_format(args.input, args.output)