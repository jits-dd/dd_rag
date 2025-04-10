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
SECTOR CLASSIFICATION GUIDE:

1. Agency
- Key Factors: Client concentration, Retainer vs project revenue mix, Gross margin, Team scalability, IP/tech assets
- Trends: Influencer platforms, content marketplaces, SMB marketing SaaS
- Description: Media sector with M&A activity focused on digital-first brands, AI content automation, and immersive formats.

2. Consumer (Apparel)
- Key Factors: Repeat rate, AOV, Inventory turnover, Gross margin, Returns %
- Trends: Fast fashion tech, athleisure D2C, sustainable fabrics
- Description: Digitally native brands with loyal communities, focus on omni-channel and inventory efficiency.

3. Consumer (Beauty & Personal Care)
- Key Factors: Hero SKUs %, Repeat rate, D2C vs retail mix, CAC/LTV
- Trends: Clean-label brands, AI personalization, Indian skin focus
- Description: Growth in clean/natural products, social commerce scaling, men's grooming rise.

4. Consumer (Food & Beverage)
- Key Factors: Unit economics, Frequency of purchase, Distribution strength
- Trends: Plant-based foods, clean snacks, regional flavor profiles
- Description: Shift toward health/conscious choices, alternative proteins, functional beverages.

5. Consumer (Footwear)
- Key Factors: Return rate, ASP, Gross margin, Offline vs online %
- Trends: Eco-friendly shoes, kids' ergonomic designs
- Description: Brands with owned manufacturing, export-ready SKUs, comfort focus.

6. Consumer (Health & Fitness)
- Key Factors: Subscription revenue %, Engagement time, CAC to LTV
- Trends: Home fitness tech, digital coaching, modern supplements
- Description: Digital-first solutions, preventive care, wearable tech adoption.

7. Consumer (Home & Household Supplies)
- Key Factors: Basket size, Distribution network, Brand loyalty
- Trends: Eco-conscious cleaning, refill models, home utility subscriptions
- Description: Hygiene-focused products, premiumization, smart appliances.

8. Consumer (Jewellery)
- Key Factors: ASP, Inventory turnover, Trust/Brand signal
- Trends: Lab-grown diamonds, silver fashion jewelry
- Description: Lightweight everyday wear, digital adoption, personalization.

9. Consumer (Personal Care)
- Key Factors: Hero SKUs %, Repeat rate, Influencer ROI
- Trends: Men's grooming, femtech, sexual wellness
- Description: Natural/Ayurvedic products, tier 2/3 market growth.

10. Fintech
- Key Factors: Take rate, Loan book quality, Regulatory moat
- Trends: Verticalized neobanks, BNPL, revenue-based financing
- Description: UPI innovation, digital lending expansion, wealth/insurtech.

11. Edtech
- Key Factors: Course completion %, CAC vs ARPU
- Trends: Job-linked upskilling, vernacular learning
- Description: Hybrid learning models, professional upskilling focus.

12. SaaS
- Key Factors: ARR, NRR, CAC payback period
- Trends: Vertical SaaS, low-code tools
- Description: Industry-specific solutions, AI integration, global demand.

13. Marketplaces
- Key Factors: GMV, Take rate, Network effects
- Trends: Social commerce, B2B agri/industrial
- Description: Vertical platforms with fintech integrations.

14. Hospitality (QSR & Restaurants)
- Key Factors: SSSG, AOV, Table turnover
- Trends: Cloud kitchens, restaurant tech
- Description: Delivery-first models, regional cuisine focus.

15. IT Services
- Key Factors: Billing rates, Utilization %, Attrition
- Trends: AI-powered MSP, cybersecurity
- Description: Digital transformation, cloud migration focus.

16. Gaming
- Key Factors: DAU/MAU, Retention %, ARPDAU
- Trends: Skill gaming, Web3 gaming
- Description: Monetization efficiency focus, cross-platform potential.

17. Healthcare & Medicine
- Key Factors: Clinical validation, Recurring revenue %
- Trends: Telehealth, elder care tech
- Description: Digitization, preventive care, tier 2/3 expansion.

18. AI / Deeptech / IoT
- Key Factors: IP defensibility, Tech team quality
- Trends: Space tech, enterprise NLP
- Description: Commercialization focus, real-time data solutions.
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
                    "content": "You are a business analyst. Extract key information about the company from the web search results."
                },
                {
                    "role": "user",
                    "content": f"Analyze this company website and describe its main business: {url}. Focus on their core services, business model, and target customers."
                }
            ],
        )

        # Extract the content and citations
        content = response.choices[0].message.content
        annotations = response.choices[0].message.annotations if hasattr(response.choices[0].message, 'annotations') else []

        return {
            "analysis": content,
            "sources": [{"url": ann.url_citation.url, "title": ann.url_citation.title}
                        for ann in annotations if ann.type == "url_citation"]
        }

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
                    Classify the company into one of these sectors:
                    {REFERENCE_DOCUMENT}
                    
                    Rules:
                    1. Choose the most specific matching sector
                    2. Explain your reasoning
                    3. If no clear match, return "Unknown"
                    """
                },
                {
                    "role": "user",
                    "content": f"Company information:\n\n{company_info}"
                }
            ],
            temperature=0.3
        )

        return response.choices[0].message.content

    except Exception as e:
        print(f"Error in sector classification: {e}")
        return None

def main():
    company_url = "https://www.urbancompany.com"

    print(f"Analyzing company: {company_url}")

    # Step 1: Extract company info using web search
    print("\nExtracting company information via web search...")
    result = analyze_company_with_web_search(company_url)

    if result and result["analysis"]:
        print("\nExtracted Information:")
        print(result["analysis"][:1000] + ("..." if len(result["analysis"]) > 1000 else ""))

        # Print sources if available
        if result["sources"]:
            print("\nSources:")
            for source in result["sources"]:
                print(f"- {source['title']}: {source['url']}")

        # Step 2: Classify sector
        print("\nClassifying sector...")
        sector = classify_sector(result["analysis"])

        print("\nClassification Result:")
        print(sector)
    else:
        print("Failed to analyze company")

if __name__ == "__main__":
    main()