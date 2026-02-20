import json
import os

class ASINFinder:
    def __init__(self, json_path="product.json"):
        self.products = {}
        # Load the JSON and create a dictionary for O(1) lookup
        if os.path.exists(json_path):
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # Map each ASIN to its specific 'content' string
                self.products = {item['asin']: item['content'] for item in data}
        else:
            print(f"Warning: {json_path} not found.")

    def get_content_by_asin(self, asin: str) -> str:
        """Returns the content string for the LLM context, or a default message."""
        return self.products.get(asin, "Product details not found.")