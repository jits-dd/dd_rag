import os
import openai
import requests
from bs4 import BeautifulSoup

import openai
import time

# Initialize OpenAI client (replace with your API key)
openai.api_key = os.getenv("OPENAI_API_KEY")

# Initialize OpenAI client
client = openai.OpenAI(api_key=openai.api_key)

# Reference document text (extracted from your PDF)
REFERENCE_DOCUMENT = """
1. Agency
2. Consumer (Apparel)
3. Consumer (Beauty & Personal Care)
4. Consumer (Food & Beverage)
5. Consumer (Footwear)
6. Consumer (Health & Fitness)
7. Consumer (Home & Household Supplies)
8. Consumer (Jewellery)
9. Consumer (Personal Care)
10. Fintech
11. Edtech
12. SaaS
13. Marketplaces
14. Hospitality (QSR & Restaurants)
15. IT Services
16. Gaming
17. Healthcare & Medicine
18. AI / Deeptech / IoT
"""

def analyze_company_with_web_search(url):
    """Use OpenAI's web search to analyze a company URL"""
    try:
        response = client.chat.completions.create(
            model="gpt-4o-search-preview",
            web_search_options={
                "search_context_size": "medium"  # Can be "low", "medium", or "high"
            },
            messages=[
                {
                    "role": "system",
                    "content": "You are a business analyst. Extract key information about the company from the web search results like - about and sector."
                },
                {
                    "role": "user",
                    "content": f"Analyze this company website and describe its main business: {url}. Focus on their core services, business model, and target customers."
                }
            ],
        )

        # Extract the content and citations
        content = response.choices[0].message.content
        return content

    except Exception as e:
        print(f"Error in web search analysis: {e}")
        return None

def classify_sector(company_info):
    """Classify the company sector based on extracted info"""
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": f"""
                    You are a business analyst who specializes in classifying 
                    the sector of a company basis the company data like - business model, about, operations etc.
                    STRICT SECTOR CLASSIFICATION RULES:
                    1. You MUST select only from these sectors:
                    {REFERENCE_DOCUMENT}
                    2. Return EXACTLY ONE of these formats:
                    - "Sector: [EXACT_SECTOR_NAME]" if matched
                    - "No matching sector found" if no clear match
                    
                    3. No explanations or additional text allowed
                    """
                },
                {
                    "role": "user",
                    "content": company_info
                }
            ],
            temperature=0.1
        )

        return response.choices[0].message.content

    except Exception as e:
        print(f"Error in sector classification: {e}")
        return None

def main():
    company_url = "https://www.acko.com/"

    print(f"Analyzing company: {company_url}")

    # Step 1: Get consie company info
    print("\nExtracting company information --")
    company_info = analyze_company_with_web_search(company_url)

    if company_info:
        print("\nCompany Description:")
        print(company_info)

        # Step 2: Strict sector classification
        print("\nClassifying sector...")
        sector = classify_sector(company_info)

        print("\nResult:")
        print(sector)
    else:
        print("Failed to analyze company")

if __name__ == "__main__":
    main()