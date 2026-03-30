import json
import argparse
from core.logger import get_logger

logger = get_logger(__name__)

def remove_spans(data):
    # If data is a dictionary, recursively check its keys
    if isinstance(data, dict):
        # Remove specific keys if present
        for key in ["cite_spans", "ref_spans", "eq_spans", "authors", "bib_entries", \
                    "year", "venue", "identifiers", "_pdf_hash", "header"]:
            data.pop(key, None)
        # Recursively apply to child dictionaries or lists
        for key, value in data.items():
            data[key] = remove_spans(value)
    # If data is a list, apply the function to each item
    elif isinstance(data, list):
        return [remove_spans(item) for item in data]
    return data


def run_pdf_process(input_json_path: str, output_json_path: str) -> None:
    with open(f'{input_json_path}') as f:
        data = json.load(f)

    cleaned_data = remove_spans(data)

    logger.info(f"[SAVED] {output_json_path}")
    with open(output_json_path, 'w') as f:
        json.dump(cleaned_data, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_json_path", type=str)
    parser.add_argument("--output_json_path", type=str)

    args = parser.parse_args()
    run_pdf_process(args.input_json_path, args.output_json_path)
