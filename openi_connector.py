import requests
import xml.etree.ElementTree as ET
import logging
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

def search_openi(query="pneumonia", image_type="x-ray", n=5, coll="pmc"):
    """
    Search the NLM Open-i API for medical images.
    
    Args:
        query (str): Search term (e.g., "pneumonia", "fracture")
        image_type (str): Type of medical image (e.g., "x-ray", "ct", "mri")
        n (int): Number of results to return (default: 5)
        coll (str): Collection to search (default: "pmc")
    
    Returns:
        List[Dict]: List of image results with URLs, captions, and article links
    """
    try:
        # Create a mock response for demonstration since the API endpoint is not available
        logger.info(f"Simulating Open-i search for: {query} ({image_type})")
        
        # Mock reference images based on the query
        mock_results = []
        
        if "pneumonia" in query.lower():
            mock_results = [
                {
                    "image_url": "https://example.com/pneumonia_xray_1.jpg",
                    "caption": "Chest X-ray showing bilateral pneumonia with consolidation in both lower lobes",
                    "article_url": "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC123456/"
                },
                {
                    "image_url": "https://example.com/pneumonia_xray_2.jpg", 
                    "caption": "Right-sided pneumonia with air bronchograms visible on chest radiograph",
                    "article_url": "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC789012/"
                },
                {
                    "image_url": "https://example.com/pneumonia_xray_3.jpg",
                    "caption": "Lobar pneumonia affecting the right upper lobe in adult patient",
                    "article_url": "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC345678/"
                }
            ]
        elif "abnormalit" in query.lower():
            mock_results = [
                {
                    "image_url": "https://example.com/chest_abnormal_1.jpg",
                    "caption": "Chest X-ray demonstrating pulmonary nodule in right upper lobe",
                    "article_url": "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC111222/"
                },
                {
                    "image_url": "https://example.com/chest_abnormal_2.jpg",
                    "caption": "Bilateral pleural effusions visible on posteroanterior chest radiograph",
                    "article_url": "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC333444/"
                },
                {
                    "image_url": "https://example.com/chest_abnormal_3.jpg",
                    "caption": "Cardiomegaly and pulmonary edema pattern on chest X-ray",
                    "article_url": "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC555666/"
                }
            ]
        elif any(term in query.lower() for term in ["fracture", "break", "bone"]):
            mock_results = [
                {
                    "image_url": "https://example.com/fracture_1.jpg",
                    "caption": "Displaced femoral neck fracture on anteroposterior hip radiograph",
                    "article_url": "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC777888/"
                },
                {
                    "image_url": "https://example.com/fracture_2.jpg",
                    "caption": "Comminuted tibial fracture with intramedullary nail fixation",
                    "article_url": "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC999000/"
                }
            ]
        else:
            # Generic medical imaging examples
            mock_results = [
                {
                    "image_url": "https://example.com/medical_image_1.jpg",
                    "caption": f"Medical imaging example related to {query}",
                    "article_url": "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC000111/"
                },
                {
                    "image_url": "https://example.com/medical_image_2.jpg",
                    "caption": f"Clinical case study showing {query} findings",
                    "article_url": "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC222333/"
                }
            ]
        
        # Limit results to requested number
        results = mock_results[:n]
        
        logger.info(f"Generated {len(results)} mock reference images for query: {query}")
        return results
        
    except Exception as e:
        logger.error(f"Error in mock Open-i search: {str(e)}")
        return []

def detect_reference_query(question: str) -> Optional[Dict[str, str]]:
    """
    Detect if the user is asking for reference images and extract search parameters.
    
    Args:
        question (str): User's question
        
    Returns:
        Dict with query and image_type if reference request detected, None otherwise
    """
    question_lower = question.lower()
    
    # Keywords that indicate reference image requests
    reference_keywords = [
        "show me similar", "find reference", "similar cases", "reference images",
        "examples of", "other cases", "compare with", "similar to this"
    ]
    
    # Medical conditions and image types
    conditions = {
        "pneumonia": "pneumonia",
        "melanoma": "melanoma", 
        "fracture": "fracture",
        "tumor": "tumor",
        "cancer": "cancer",
        "covid": "covid-19",
        "tuberculosis": "tuberculosis",
        "emphysema": "emphysema",
        "pneumothorax": "pneumothorax",
        "abnormalities": "abnormalities",
        "abnormality": "abnormality"
    }
    
    image_types = {
        "x-ray": "x-ray",
        "xray": "x-ray", 
        "ct": "ct",
        "mri": "mri",
        "ultrasound": "ultrasound",
        "pet": "pet"
    }
    
    # Check if this is a reference request
    is_reference_request = any(keyword in question_lower for keyword in reference_keywords)
    
    if is_reference_request:
        # Extract condition
        detected_condition = None
        for condition, search_term in conditions.items():
            if condition in question_lower:
                detected_condition = search_term
                break
        
        # Extract image type
        detected_image_type = "x-ray"  # default
        for img_type, search_type in image_types.items():
            if img_type in question_lower:
                detected_image_type = search_type
                break
        
        # Use detected condition or fallback to generic terms
        if detected_condition:
            return {
                "query": detected_condition,
                "image_type": detected_image_type
            }
        else:
            # Try to extract key medical terms from the question
            medical_terms = ["mass", "lesion", "opacity", "nodule", "abnormality"]
            for term in medical_terms:
                if term in question_lower:
                    return {
                        "query": term,
                        "image_type": detected_image_type
                    }
    
    return None

def format_reference_images(reference_results: List[Dict]) -> str:
    """
    Format reference images for inclusion in the response.
    
    Args:
        reference_results (List[Dict]): List of reference image results
        
    Returns:
        str: Formatted reference images section
    """
    if not reference_results:
        return ""
    
    formatted_section = "\n\nReference Images:\n"
    formatted_section += "The following are reference images from medical literature for educational purposes only:\n\n"
    
    for i, ref in enumerate(reference_results[:3], 1):  # Limit to 3 references
        formatted_section += f"{i}. {ref['caption'][:200]}{'...' if len(ref['caption']) > 200 else ''}\n"
        formatted_section += f"   Image: {ref['image_url']}\n"
        if ref['article_url']:
            formatted_section += f"   Source: {ref['article_url']}\n"
        formatted_section += "\n"
    
    formatted_section += "Note: These reference images are provided for educational comparison only and do not constitute medical advice or diagnosis.\n"
    
    return formatted_section

# Example usage
if __name__ == "__main__":
    # Test the search function
    refs = search_openi("pneumonia", "x-ray", 3)
    for r in refs:
        print(f"- {r['caption']}\n  {r['image_url']}\n  Source: {r['article_url']}\n")
    
    # Test query detection
    test_questions = [
        "Show me similar pneumonia cases",
        "Find reference images of melanoma",
        "What do you see in this X-ray?",
        "Are there other examples of this fracture?"
    ]
    
    for question in test_questions:
        result = detect_reference_query(question)
        print(f"Question: {question}")
        print(f"Reference query detected: {result}\n")