def process_image_with_captions(raw_chunks, use_gemini=True):
    """
    Extract images from document chunks, identify captions and generate descriptions.
    Args:
        raw_chunks: List of document elements from unstructured.partition_pdf
        use_gemini: Whether to use Gemini for image captioning (default: True)
    Returns:
        List of dictionaries with image data, captions, and generated descriptions
        encountered_errors: List of dictionaries containing any errors encountered during processing
    """
    import base64
    import os
    
    import google.generativeai as genai
    from dotenv import load_dotenv
    from unstructured.documents.elements import FigureCaption, Image
    
    load_dotenv()
    
    # Configure Gemini API
    if use_gemini:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY is not set in the environment variables.")
        genai.configure(api_key=api_key)
        
    # Extract images and their captions
    processed_images = []
    encountered_errors = []
    
    for idx, chunk in enumerate(raw_chunks):
        if isinstance(chunk, Image):
            # Check if next element is a figure caption
            if (idx + 1) < len(raw_chunks) and isinstance(raw_chunks[idx+1], FigureCaption):
                caption = raw_chunks[idx+1].text
            else:
                caption = "No caption available"
            
            # store image data
            image_data = {
                "caption" : caption,
                "image_text" : chunk.text if hasattr(chunk, "text") else "",
                "base64_image" : chunk.metadata.image_base64,
                "content" : (chunk.text if hasattr(chunk, "text") else ""), # Fallback content
                "content_type" : "image",
                "filename" : (chunk.metadata.filename if hasattr(chunk, "metadata") else ""),
            }    
            
            error_data = {
                "error" : None,
                "error_message" : None
            }
            
            # Generate description if requested
            if use_gemini:
                try:
                    image_binary = base64.b64decode(chunk.metadata.image_base64)
                    
                    # use gemini model for image description
                    model = genai.GenerativeModel("gemini-2.5-flash")
                    
                    prompt = (
                        f"Generate a comprehensive and detailed description of this image from a technical document abot Retrieval-Augemnted generation. \n\n"
                        f"CONTEXT INFROMATION:\n"
                        f"- Caption : {caption}\n"
                        f"- Text Extracted from image : {chunk.text if hasattr(chunk, 'text') else 'No text'}\n\n"
                        f"DESCRIPTION REQUIREMENTS:\n"
                        f"1. begin with a clear overview of what the image shows (diagram, chart, architecture, etc.)\n"
                        f"2. If it's a diagram or flowchart : describe components, connections, data flow direction, and system architecture\n"
                        f"3. If it's a chart or flowchart : explain axes, trends, key data points and significance\n"
                        f"4. Explain technical terminology and abbreviations that appear in the image\n"
                        f"5. Interpret how this visual relates to RAG concepts and implementation\n"
                        f"6. Include any numerical data, performance metrics, or comparative results shown\n"
                        f"7. target length : 150-300 words for complex diagrams, 100-150 words for simple images\n\n"
                        f"Focus on providing information that would be valuable in a technical context for someone implementing or researching RAG systems."
                    )
                    
                    response = model.generate_content(
                        [prompt, {"mime_type" : "image/jpg", "data":image_binary}]
                    )
                    
                    image_data["content"] = response.text
                    
                except Exception as e:
                    print(f"Warning: Error generating description: {str(e)}")
                    
                    error_data["error"] = str(e)
                    error_data["error_message"] = (
                        "Error generating description with Gemini."
                    )
                    encountered_errors.append(error_data)
                
            processed_images.append(error_data)
    
    print(f"Processed {len(processed_images)} images with captions and descriptions")
    print(f"Errors encountered : {len(encountered_errors)}")
    return processed_images, encountered_errors


def process_tables_with_descriptions(raw_chunks, use_gemini = True, use_ollama=True):
    """Process tables with descriptions using Gemini or Ollama"""
    import os

    import google.generativeai as genai
    import requests
    from dotenv import load_dotenv
    from unstructured.documents.elements import Table

    load_dotenv()

    # Configure Gemini API
    if use_gemini:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY is not set in the environment variables.")
        genai.configure(api_key=api_key)
        
        
        
        
        
if __name__ == "__main__":
    from unstructured.partition.pdf import partition_pdf
    
    # Example usage
    pdf_path = r"D:\Generative_AI\Projects\multi_doc_deepak_mittal\survey_llm.pdf"
    # raw_chunks = partition_pdf(pdf_path)
    
    
    raw_chunks = partition_pdf(
    filename = pdf_path,
    strategy = "hi_res",
    infer_table_structure = True,
    extract_image_block_to_payload = True,
    extract_images_in_pdf=True, # very imp to get image.base64 of images
    extract_images_block_types = ["Image", "Figure", "Table"]
        )
    
    processed_images = process_image_with_captions(raw_chunks, use_gemini=True)
        
        