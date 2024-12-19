import json
import os
from pathlib import Path
from typing import List, Dict, Any, Union
import streamlit as st
import ollama
from langchain_core.documents import Document

# Previous imports and constants remain the same...

class ProductProcessor:
    def __init__(self):
        self.llm = LLMProcessor()
    
    def process_json_document(self, json_content: str) -> Dict[str, Any]:
        """Process JSON document and add AI summaries for each product."""
        try:
            # Parse JSON content
            products = json.loads(json_content)
            
            # Ensure we have a list of products
            if not isinstance(products, list):
                raise ValueError("JSON content must be a list of products")
            
            # Process each product
            for i, product in enumerate(products, 1):
                if not isinstance(product, dict):
                    continue
                    
                # Show progress
                st.progress(i / len(products), f"Processing product {i} of {len(products)}")
                    
                # Generate AI summary for the product
                summary = self._generate_product_summary(product)
                product['ai_summary'] = summary
            
            return products
            
        except json.JSONDecodeError as e:
            st.error(f"Invalid JSON format: {str(e)}")
            raise
        except Exception as e:
            st.error(f"Error processing products: {str(e)}")
            raise
    
    def _generate_product_summary(self, product: Dict[str, Any]) -> List[str]:
        """Generate an AI summary for a single product."""
        try:
            product_info = json.dumps(product, indent=2)
            
            response = ollama.chat(
                model="llama3.2:3b",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a product analysis assistant. Create concise, bullet-point summaries of product features.",
                    },
                    {
                        "role": "user",
                        "content": SUMMARY_PROMPT.format(product_info=product_info),
                    },
                ],
            )
            
            summary_text = response['message']['content']
            bullet_points = [
                point.strip().lstrip('•-*').strip()
                for point in summary_text.split('\n')
                if point.strip() and point.strip()[0] in '•-*'
            ]
            
            return bullet_points
            
        except Exception as e:
            st.error(f"Error generating summary for product: {str(e)}")
            return ["Error generating summary"]

class FolderProcessor:
    def __init__(self, folder_path: str):
        self.folder_path = Path(folder_path)
        self.product_processor = ProductProcessor()
    
    def process_folder(self) -> Dict[str, Any]:
        """Process all JSON files in the specified folder."""
        try:
            results = {}
            json_files = list(self.folder_path.glob("*.json"))
            
            if not json_files:
                st.warning(f"No JSON files found in {self.folder_path}")
                return results
            
            for json_file in json_files:
                st.write(f"Processing {json_file.name}...")
                
                try:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        json_content = f.read()
                        
                    processed_products = self.product_processor.process_json_document(json_content)
                    
                    # Save processed file
                    output_path = self.folder_path / f"processed_{json_file.name}"
                    with open(output_path, 'w', encoding='utf-8') as f:
                        json.dump(processed_products, f, indent=2)
                    
                    results[json_file.name] = {
                        'status': 'success',
                        'output_path': str(output_path),
                        'product_count': len(processed_products)
                    }
                    
                except Exception as e:
                    results[json_file.name] = {
                        'status': 'error',
                        'error': str(e)
                    }
                    st.error(f"Error processing {json_file.name}: {str(e)}")
            
            return results
            
        except Exception as e:
            st.error(f"Error processing folder: {str(e)}")
            raise

class RAGApplication:
    def __init__(self):
        self.product_processor = ProductProcessor()
        self.vector_store = VectorStore()
    
    def process_json_file(self, json_content: Union[str, bytes]) -> None:
        """Process JSON content and add AI summaries."""
        try:
            # Convert bytes to string if needed
            if isinstance(json_content, bytes):
                json_content = json_content.decode('utf-8')
            
            # Process products and add summaries
            processed_products = self.product_processor.process_json_document(json_content)
            
            # Create download button for processed JSON
            processed_json = json.dumps(processed_products, indent=2)
            st.download_button(
                label="Download Processed JSON",
                data=processed_json,
                file_name="processed_products.json",
                mime="application/json"
            )
            
            # Display sample results
            st.success("Products processed successfully!")
            st.subheader("Sample Results")
            for idx, product in enumerate(processed_products[:3], 1):
                st.write(f"\nProduct {idx}:")
                st.json(product)
                
                st.write("AI Summary:")
                for point in product['ai_summary']:
                    st.write(f"• {point}")
                
                if idx < len(processed_products):
                    st.markdown("---")
            
            if len(processed_products) > 3:
                st.info(f"{len(processed_products) - 3} more products processed. Download the JSON file to see all results.")
                
        except Exception as e:
            st.error(f"Error processing JSON file: {str(e)}")
            raise

def main():
    st.set_page_config(page_title="Product Summary Generator")
    app = RAGApplication()

    st.title("🏷️ Product Summary Generator")
    st.write("Upload a JSON file or process JSON files from a folder to generate AI summaries.")

    # Create tabs for different input methods
    upload_tab, folder_tab = st.tabs(["Upload JSON", "Process Folder"])

    # Upload JSON tab
    with upload_tab:
        uploaded_file = st.file_uploader(
            "**📑 Upload JSON file**",
            type=["json"],
            accept_multiple_files=False
        )

        if uploaded_file and st.button("⚡️ Process Uploaded File"):
            with st.spinner("Processing uploaded file..."):
                app.process_json_file(uploaded_file.read())

    # Process folder tab
    with folder_tab:
        folder_path = st.text_input(
            "**📁 Enter folder path**",
            placeholder="e.g., /path/to/json/files"
        )
        
        if folder_path and st.button("🔄 Process Folder"):
            if not os.path.exists(folder_path):
                st.error("Folder path does not exist!")
            else:
                with st.spinner("Processing files in folder..."):
                    folder_processor = FolderProcessor(folder_path)
                    results = folder_processor.process_folder()
                    
                    # Display results summary
                    st.subheader("Processing Results")
                    for filename, result in results.items():
                        if result['status'] == 'success':
                            st.success(f"✅ {filename}: Processed {result['product_count']} products")
                            st.write(f"Output saved to: {result['output_path']}")
                        else:
                            st.error(f"❌ {filename}: {result['error']}")

if __name__ == "__main__":
    main()
