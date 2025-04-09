import os
import json
import requests
import argparse
import time
import logging
from typing import List, Dict, Optional, Any
from dotenv import load_dotenv
from firecrawl import FirecrawlApp
import google.generativeai as genai
import hashlib
import configparser
import sys

# gemini-2.0-flash-thinking-exp-01-21
# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger('job_hunt_automator')

# ANSI color codes for terminal output
class Colors:
    CYAN = '\033[96m'
    YELLOW = '\033[93m'
    GREEN = '\033[92m'
    RED = '\033[91m'
    MAGENTA = '\033[95m'
    BLUE = '\033[94m'
    RESET = '\033[0m'

# Load environment variables from .env file
load_dotenv()

# Configuration
CONFIG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.ini')

def load_config():
    """Load configuration from config.ini file or create default if not exists"""
    config = configparser.ConfigParser()

    if not os.path.exists(CONFIG_FILE):
        logger.warning(f"Config file not found at {CONFIG_FILE}. Creating default.")
        # Create default config reflecting Gemini
        config['API'] = {
            'gemini_model': 'gemini-2.0-flash-thinking-exp-01-21', # Default to Gemini
            'api_timeout': '60',
            'max_retries': '3',
            'retry_delay': '2'
        }
        config['EXTRACTION'] = {
            'max_urls_to_process': '3',
            'description_max_length': '1900' # Max length for Notion rich text (2000 limit)
        }
        try:
            with open(CONFIG_FILE, 'w') as f:
                config.write(f)
            logger.info(f"Default config file created at {CONFIG_FILE}")
        except IOError as e:
            logger.error(f"Unable to create config file: {e}")

    else:
        config.read(CONFIG_FILE)

    return config

CONFIG = load_config()

# Retrieve API keys and configuration from environment variables
FIRECRAWL_API_KEY = os.getenv("FIRECRAWL_API_KEY")
NOTION_API_KEY = os.getenv("NOTION_API_KEY")
NOTION_DATABASE_ID = os.getenv("NOTION_DATABASE_ID")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# --- Global Variables and Clients ---
gemini_model = None
app = None

# API settings from config
API_TIMEOUT = CONFIG.getint('API', 'api_timeout', fallback=60)
MAX_RETRIES = CONFIG.getint('API', 'max_retries', fallback=3)
RETRY_DELAY = CONFIG.getint('API', 'retry_delay', fallback=2)
MAX_URLS_TO_PROCESS = CONFIG.getint('EXTRACTION', 'max_urls_to_process', fallback=3)
DESCRIPTION_MAX_LENGTH = CONFIG.getint('EXTRACTION', 'description_max_length', fallback=1900)
GEMINI_MODEL_NAME = CONFIG.get('API', 'gemini_model', fallback='gemini-2.0-flash-thinking-exp-01-21')


# --- Initialize Clients ---
try:
    if GOOGLE_API_KEY:
        genai.configure(api_key=GOOGLE_API_KEY)
        gemini_model = genai.GenerativeModel(GEMINI_MODEL_NAME)
        logger.info(f"{Colors.CYAN}Configured Google Generative AI with model: {GEMINI_MODEL_NAME}{Colors.RESET}")
    else:
         logger.warning(f"{Colors.YELLOW}GOOGLE_API_KEY not found in environment variables. AI features will be disabled.{Colors.RESET}")
except Exception as e:
    logger.error(f"{Colors.RED}Failed to configure Google Generative AI: {e}{Colors.RESET}")
    gemini_model = None # Ensure it's None on failure

try:
    if FIRECRAWL_API_KEY:
        app = FirecrawlApp(api_key=FIRECRAWL_API_KEY)
        logger.info(f"{Colors.CYAN}Configured FirecrawlApp.{Colors.RESET}")
    else:
        logger.warning(f"{Colors.YELLOW}FIRECRAWL_API_KEY not found. Web scraping features will be disabled.{Colors.RESET}")
except Exception as e:
     logger.error(f"{Colors.RED}Failed to configure FirecrawlApp: {e}{Colors.RESET}")
     app = None


# --- Helper Functions ---
def retry_api_call(func, *args, **kwargs):
    """Retry an API call with exponential backoff"""
    # Add timeout to kwargs if not present, applicable for requests.post
    if func == requests.post and 'timeout' not in kwargs:
         kwargs['timeout'] = API_TIMEOUT

    for attempt in range(MAX_RETRIES):
        try:
            return func(*args, **kwargs)
        except requests.exceptions.Timeout as e:
             logger.warning(f"API call timed out: {str(e)}")
             # Handle timeout specifically if needed, e.g., shorter retry delay?
             wait_time = RETRY_DELAY * (2 ** attempt)
        except requests.exceptions.RequestException as e: # Catch requests specific errors
             logger.warning(f"Network/Request error: {str(e)}")
             wait_time = RETRY_DELAY * (2 ** attempt)
        except Exception as e: # Catch other errors (GenAI, Firecrawl, etc.)
            # Check for potential rate limit errors if possible (depends on library specifics)
            # e.g., if 'rate limit' in str(e).lower(): wait_time *= 2
            logger.warning(f"API call failed: {str(e)} (Attempt {attempt + 1}/{MAX_RETRIES})")
            wait_time = RETRY_DELAY * (2 ** attempt)

        if attempt < MAX_RETRIES - 1:
            logger.info(f"Retrying in {wait_time:.2f} seconds...")
            time.sleep(wait_time)
        else:
            logger.error(f"API call failed after {MAX_RETRIES} attempts: {str(e)}")
            raise # Re-raise the last exception

def parse_gemini_json(response_text):
    """Helper to parse JSON from Gemini responses, handling potential markdown wrappers"""
    if not response_text:
        return None
        
    text = response_text.strip()
    if text.startswith("```json"): 
        text = text[7:].strip()
    elif text.startswith("```"): 
        text = text[3:].strip()
    
    if text.endswith("```"):
        text = text[:-3].strip()
        
    if not text:
        return None
        
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        logger.error(f"Error parsing JSON from Gemini: {e}")
        return None

# --- Core Logic Functions ---

def generate_search_parameter(objective):
    """Generate a search keyword based on job objective using Gemini API"""
    if not gemini_model:
        logger.error(f"{Colors.RED}Gemini model not available for search parameter generation.{Colors.RESET}")
        return objective.split()[0] # Basic fallback

    map_prompt = f"""
    Analyze the job objective: "{objective}".
    Identify 1-3 core keywords suitable for searching job boards or websites for relevant postings.
    Focus on nouns and technologies. Example: If objective is "Remote Senior Python Developer specializing in APIs", keywords could be "Python Developer API".
    Respond ONLY with the space-separated keywords. Do not add explanations or formatting.
    """
    logger.info(f"{Colors.YELLOW}Generating search keyword using Gemini...{Colors.RESET}")

    try:
        response = retry_api_call(
            gemini_model.generate_content,
            map_prompt,
            generation_config=genai.types.GenerationConfig(temperature=0.1) # Lower temp for focused keywords
        )
        keyword = response.text.strip()
        if not keyword: # Handle empty response
             raise ValueError("Gemini returned empty keyword.")
        logger.info(f"{Colors.GREEN}Generated search keyword(s): {keyword}{Colors.RESET}")
        return keyword
    except Exception as e:
        logger.error(f"{Colors.RED}Failed to generate search parameter with Gemini: {str(e)}{Colors.RESET}")
        # More robust fallback if objective is short
        parts = objective.split()
        return " ".join(parts[:2]) if len(parts)>1 else parts[0] if parts else "job"


def map_website_with_keyword(url, keyword):
    """Map website to find candidate job URLs using the keyword"""
    if not app:
        logger.error(f"{Colors.RED}FirecrawlApp not available for website mapping.{Colors.RESET}")
        return []

    logger.info(f"{Colors.YELLOW}Mapping website {url} with search parameter: '{keyword}'{Colors.RESET}")
    # Firecrawl map_url doesn't have a direct search param, we use scrape+extract logic later
    # Here, we just get potential links from the entry URL itself using scrape
    # Or, if Firecrawl had a site-wide map feature, we'd use it here.
    # Let's assume we first scrape the entry URL to find links ON THAT PAGE.

    try:
        # Scrape the initial URL to find links on the page
        # Increase crawl depth slightly if needed, but be mindful of cost/time
        # Pass scrape options directly, not nested under pageOptions
        scrape_params = {'includeHtml': False} # Let's only specify includeHtml for now # Get all links
        scrape_result = retry_api_call(app.scrape_url, url, params=scrape_params)

        if not scrape_result or not scrape_result.get('markdown'):
             logger.warning(f"Could not scrape initial content from {url}")
             return []

        page_markdown = scrape_result.get('markdown')

        # Use Gemini (or regex) to find potential job-related links within the markdown
        # This replaces the potentially non-existent Firecrawl 'map' search
        find_links_prompt = f"""
        Analyze the following Markdown content from the URL {url}.
        The goal is to find job postings related to: "{keyword}".

        Identify all URLs within the content that likely lead to job listing pages, career sections, or specific job postings.
        Look for patterns like '/jobs', '/careers', '/openings', job titles in links, etc.

        Respond ONLY with a valid JSON array of unique URL strings found.
        Example: ["https://example.com/careers/job1", "https://example.com/jobs?id=2"]
        If no relevant links are found, respond with an empty JSON array: `[]`.
        Do not include explanations or markdown formatting.

        Markdown Content (first 15000 chars):
        --- START CONTENT ---
        {page_markdown[:15000]}
        --- END CONTENT ---
        """

        logger.info(f"{Colors.YELLOW}Using Gemini to identify potential job links on page...{Colors.RESET}")
        if not gemini_model:
             logger.error(f"{Colors.RED}Gemini model not available to find links.{Colors.RESET}")
             return [] # Cannot proceed without LLM here

        generation_config=genai.types.GenerationConfig(response_mime_type="application/json")
        response = retry_api_call(
            gemini_model.generate_content,
            find_links_prompt,
            generation_config=generation_config
        )
        response_text = response.text.strip()
        links = parse_gemini_json(response_text) or []
        
        if not isinstance(links, list):
             logger.error(f"{Colors.RED}Invalid JSON received from Gemini for link finding (expected list){Colors.RESET}")
             links = []

        # Basic validation/normalization of links (e.g., ensure they are absolute URLs)
        from urllib.parse import urljoin
        base_url = url
        valid_links = set() # Use a set for automatic deduplication
        for link in links:
             if isinstance(link, str) and link.startswith(('http://', 'https://')):
                  valid_links.add(link)
             elif isinstance(link, str) and link.startswith('/'):
                  valid_links.add(urljoin(base_url, link))
             # Add more robust link validation if needed

        links = list(valid_links)
        logger.info(f"{Colors.GREEN}Found {len(links)} potential job links on the page.{Colors.RESET}")
        return links

    except json.JSONDecodeError as e:
        logger.error(f"{Colors.RED}Error decoding JSON from Gemini link finding: {e}{Colors.RESET}")
        return []
    except Exception as e:
        logger.error(f"{Colors.RED}Error during website mapping/link extraction: {e}{Colors.RESET}", exc_info=True)
        return []


def rank_candidate_urls(links, objective):
    """Rank candidate URLs by relevance to job objective using Gemini API"""
    if not gemini_model:
        logger.error(f"{Colors.RED}Gemini model not available for ranking.{Colors.RESET}")
        return links[:MAX_URLS_TO_PROCESS] if links else []
    if not links:
        return []

    # Limit number of links sent to Gemini to avoid large prompts/costs
    links_to_rank = links[:50] # Rank up to 50 links

    rank_prompt = f"""
    Analyze the following list of URLs based on the job objective: "{objective}".
    Your goal is to select the top {MAX_URLS_TO_PROCESS} URLs most likely to contain SPECIFIC job postings relevant to the objective. Prioritize URLs that seem to point to individual job descriptions over general career pages if possible, but include career pages if they seem highly relevant.

    URLs to analyze:
    {json.dumps(links_to_rank, indent=2)}

    Respond ONLY with a valid JSON array containing up to {MAX_URLS_TO_PROCESS} objects, ordered by relevance (most relevant first). Each object must have these keys:
    - "url": The full URL string.
    - "relevance_score": A number between 0 and 100 indicating relevance.
    - "reason": A brief string explaining the ranking.

    DO NOT include any text before or after the JSON array.
    DO NOT use markdown formatting (like ```json).
    If no URLs seem relevant, return an empty JSON array `[]`.
    """
    logger.info(f"{Colors.YELLOW}Ranking {len(links_to_rank)} candidate URLs using Gemini...{Colors.RESET}")

    try:
        generation_config=genai.types.GenerationConfig(
            response_mime_type="application/json",
            temperature=0.2 # Lower temp for more deterministic ranking
        )

        response = retry_api_call(
            gemini_model.generate_content,
            rank_prompt,
            generation_config=generation_config
        )
        ranked_results = parse_gemini_json(response.text) or []
        
        if not isinstance(ranked_results, list):
             raise ValueError("Invalid JSON structure received from Gemini for ranking (expected a list).")

        # Extract top URLs based on the ranking order, up to the max limit
        top_urls = [entry["url"] for entry in ranked_results[:MAX_URLS_TO_PROCESS] if isinstance(entry, dict) and "url" in entry]

        logger.info(f"{Colors.GREEN}Top {len(top_urls)} ranked URLs (Gemini): {top_urls}{Colors.RESET}")
        return top_urls
    except (json.JSONDecodeError, ValueError, Exception) as e:
        logger.error(f"{Colors.RED}Error ranking URLs with Gemini: {str(e)}{Colors.RESET}")
        # Fallback to first MAX_URLS_TO_PROCESS links if ranking fails
        logger.info(f"{Colors.YELLOW}Falling back to using first {MAX_URLS_TO_PROCESS} found links.{Colors.RESET}")
        return links[:MAX_URLS_TO_PROCESS] if links else []


def extract_job_data_from_page(url, objective):
    """Extract job posting data from a webpage using Gemini API"""
    if not gemini_model:
        logger.error(f"{Colors.RED}Gemini model not available for extraction.{Colors.RESET}")
        return None
    if not app:
        logger.error(f"{Colors.RED}FirecrawlApp not available for scraping.{Colors.RESET}")
        return None

    logger.info(f"{Colors.YELLOW}Attempting to scrape and extract job data from: {url}{Colors.RESET}")

    try:
        # Step 1: Scrape the page content using Firecrawl
        # Prioritize main content extraction if possible
        # Pass scrape options directly, not nested under pageOptions
        scrape_params = {'onlyMainContent': True, 'includeHtml': False}
        scrape_result = retry_api_call(app.scrape_url, url, params=scrape_params)
        page_content = scrape_result.get('markdown', scrape_result.get('content', ''))

        if not page_content or len(page_content) < 50:
             logger.warning(f"{Colors.YELLOW}Skipping URL due to insufficient content: {url}{Colors.RESET}")
             # Return empty list, indicating no jobs found on this page
             return []

        # Step 2: Use Gemini to extract structured data
        extract_prompt = f"""
        Analyze the following web page content obtained from {url}.
        The primary job search objective is: "{objective}"

        Page Content (Markdown/Text):
        --- START CONTENT ---
        {page_content[:15000]}
        --- END CONTENT ---

        Instructions:
        1. Identify if the content contains one or more job listings relevant to the objective: "{objective}".
        2. If relevant job listings are found, extract the following fields for EACH distinct job:
           - "job_title": The title of the job position (string).
           - "company": The name of the company offering the job (string, infer if possible, else "Unknown").
           - "location": Location (e.g., "Remote", "City, ST", "Hybrid") (string, "Unknown" if not found).
           - "job_description": A concise summary (max 300-400 words) of responsibilities/requirements (string).
           - "application_url": The direct URL to apply, if found. If not found, use the source URL "{url}" (string).
        3. Format the output ONLY as a valid JSON array. Each element of the array should be a JSON object representing one job.
        4. Ensure all values are strings. Use "Unknown" if information for a field cannot be found. Clean any HTML/markdown from extracted text.
        5. If NO relevant job listings matching the objective are found on the page, respond ONLY with an empty JSON array: `[]`.

        Respond ONLY with the JSON array. Do not include explanations or markdown formatting.
        """

        logger.info(f"{Colors.YELLOW}Extracting job data using Gemini from: {url}{Colors.RESET}")

        generation_config=genai.types.GenerationConfig(
            response_mime_type="application/json"
        )

        response = retry_api_call(
            gemini_model.generate_content,
            extract_prompt,
            generation_config=generation_config
        )
        extracted_data = parse_gemini_json(response.text) or []
        
        if not isinstance(extracted_data, list):
            logger.error(f"{Colors.RED}Invalid JSON structure received from Gemini (expected list): {url}{Colors.RESET}")
            return None # Indicate error

        if not extracted_data:
            logger.info(f"{Colors.YELLOW}No relevant job listings found by Gemini on URL: {url}{Colors.RESET}")
            return []

        processed_jobs = []
        for job in extracted_data:
            if not isinstance(job, dict): # Ensure element is a dictionary
                 logger.warning(f"Skipping invalid entry in extracted data (not a dict): {job}")
                 continue
            if not all(k in job for k in ["job_title", "company", "location", "job_description", "application_url"]):
                 logger.warning(f"Skipping job entry with missing keys: {job.get('job_title', 'N/A')} from {url}")
                 continue

            if len(job.get("job_description", "")) > DESCRIPTION_MAX_LENGTH:
                job["job_description"] = job["job_description"][:DESCRIPTION_MAX_LENGTH] + "..."
            if not job.get("application_url"):
                job["application_url"] = url

            job_string = f"{job.get('job_title', '')}-{job.get('company', '')}-{job.get('location', '')}"
            job["job_id"] = hashlib.md5(job_string.encode()).hexdigest()
            processed_jobs.append(job)

        logger.info(f"{Colors.GREEN}Extracted {len(processed_jobs)} job listings via Gemini from URL: {url}{Colors.RESET}")
        return processed_jobs

    except json.JSONDecodeError as e:
        logger.error(f"{Colors.RED}Error decoding JSON for {url}: {e}{Colors.RESET}")
        return None
    except Exception as e:
        logger.error(f"{Colors.RED}Error processing URL {url} for extraction: {e}{Colors.RESET}", exc_info=True)
        return None


# --- Notion Integration Functions ---

# --- Notion Integration Functions ---

def get_existing_jobs_from_notion():
    """Retrieve existing jobs from Notion database to avoid duplicates"""
    if not NOTION_API_KEY or not NOTION_DATABASE_ID:
         logger.error(f"{Colors.RED}Notion API Key or Database ID missing. Cannot fetch existing jobs.{Colors.RESET}")
         return {}

    url = f"https://api.notion.com/v1/databases/{NOTION_DATABASE_ID}/query"
    headers = {
        "Authorization": f"Bearer {NOTION_API_KEY}",
        "Content-Type": "application/json",
        "Notion-Version": "2022-06-28"
    }

    existing_jobs = {}
    has_more = True
    start_cursor = None
    page_count = 0
    logger.info(f"{Colors.CYAN}Fetching existing job IDs from Notion...{Colors.RESET}")

    try:
        while has_more:
            payload = {}
            if start_cursor:
                payload["start_cursor"] = start_cursor

            response = retry_api_call(
                requests.post,
                url,
                headers=headers,
                json=payload # Use payload which might contain start_cursor
            )
            response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
            result = response.json()
            page_count += 1

            for page in result.get("results", []):
                 props = page.get("properties", {})
                 
                 # --- Start of Changed Block ---
                 # Adapt based on actual Notion property names and types from your screenshots
                 company_prop = props.get("Company", {}).get("title", []) # Assuming 'Company' is the Title property
                 position_prop = props.get("Position", {}).get("rich_text", []) # Assuming 'Position' is Rich Text
                 location_prop = props.get("Location", {}).get("rich_text", []) # Assuming 'Location' is Rich Text

                 # Extract the text content
                 company = company_prop[0].get("text", {}).get("content", "") if company_prop else ""
                 position = position_prop[0].get("text", {}).get("content", "") if position_prop else ""
                 location = location_prop[0].get("text", {}).get("content", "") if location_prop else ""

                 # Generate the job_string using the correct fields
                 if position or company or location: # Only hash if we have some data
                     # Use the fields corresponding to the script's extraction order (title, company, location)
                     # which map to Notion's (Position, Company, Location)
                     job_string = f"{position}-{company}-{location}" # NOTE: This must match the fields used for ID generation in extract_job_data_from_page
                     job_id = hashlib.md5(job_string.encode()).hexdigest()
                     existing_jobs[job_id] = page["id"] # Store page ID
                 # --- End of Changed Block ---

            has_more = result.get("has_more", False)
            start_cursor = result.get("next_cursor")
            if has_more:
                 logger.info(f"Fetched page {page_count}, more results exist...")
        time.sleep(0.5) # Brief pause between paginated requests

        logger.info(f"{Colors.CYAN}Found {len(existing_jobs)} existing job IDs in Notion after fetching {page_count} pages.{Colors.RESET}")
        return existing_jobs
    except requests.exceptions.RequestException as e:
         logger.error(f"{Colors.RED}Network error fetching Notion jobs: {e}{Colors.RESET}")
         return {} # Return empty on error
    except Exception as e:
        logger.error(f"{Colors.RED}Error retrieving existing jobs from Notion: {e}{Colors.RESET}", exc_info=True)
        return {}


def add_job_to_notion(job):
    """Add a single job posting to Notion database"""
    if not NOTION_API_KEY or not NOTION_DATABASE_ID:
         logger.error(f"{Colors.RED}Notion API Key or Database ID missing. Cannot add job.{Colors.RESET}")
         return False

    url = "https://api.notion.com/v1/pages"
    headers = {
        "Authorization": f"Bearer {NOTION_API_KEY}",
        "Content-Type": "application/json",
        "Notion-Version": "2022-06-28"
    }

    # Ensure required fields are present and handle potential None values gracefully
    # These keys come from the 'extract_job_data_from_page' function's output
    job_title = job.get("job_title", "N/A") # This will map to Notion's "Position"
    company = job.get("company", "N/A")     # This will map to Notion's "Company" (Title)
    location = job.get("location", "N/A")   # This will map to Notion's "Location"
    application_url = job.get("application_url", "") # This will map to Notion's "Job URL"
    job_description = job.get("job_description", "N/A") # This will map to Notion's "Notes"

    # Truncate description before creating the payload if it will go into 'Notes'
    if len(job_description) > DESCRIPTION_MAX_LENGTH:
        logger.warning(f"Truncating description for Notion 'Notes' field for job: {job_title}")
        job_description = job_description[:DESCRIPTION_MAX_LENGTH] + "..."

    # --- Start of Changed Block ---
    # Define Notion page properties based on your ACTUAL database schema
    page_data = {
        "parent": {"database_id": NOTION_DATABASE_ID},
        "properties": {
            # --- Mapped Properties ---
            "Company": { # This MUST be your 'Title' property in Notion
                "title": [{"text": {"content": company}}]
            },
            "Position": { # Mapped from script's 'job_title' - Assumed Rich Text type
                         # Change "rich_text" to "title" if "Position" is actually your Title property
                "rich_text": [{"text": {"content": job_title}}]
            },
            "Location": { # Mapped from script's 'location' - Assumed Rich Text type
                "rich_text": [{"text": {"content": location}}]
            },
            "Notes": { # Mapped from script's 'job_description' - Assumed Rich Text type
                "rich_text": [{"text": {"content": job_description}}]
            },
            "Job URL": { # Mapped from script's 'application_url' - Assumed URL type
                       # Change "url" key if "Job URL" is not a URL type property
                "url": application_url if application_url else None # Use None if URL is empty string to avoid API error
            },

            # --- Optional: Add Default Status ---
            # Uncomment this block if you want to automatically set the status for new entries.
            # Make sure "Not Started" (or your desired default) is an EXACT option
            # in your "Status" Select property in Notion.
            # "Status": {
            #    "select": { "name": "Not Started" }
            # },

            # --- Other Properties (Not currently populated by script but exist in Notion) ---
            # These are placeholders showing how you *could* populate them if you extracted the data.
            # The script currently does NOT extract Salary, Contact Person, Next Steps, or set Date Applied.
            # "Date Applied": { "date": None }, # e.g., { "date": { "start": "2025-04-10" } }
            # "Salary": { "rich_text": [] }, # e.g., { "rich_text": [{"text": {"content": "$100k"}}] }
            # "Contact Person": { "rich_text": [] },
            # "Next Steps": { "rich_text": [] },
        }
    }
    # --- End of Changed Block ---

    try:
        logger.debug(f"Attempting to add job to Notion with payload: {json.dumps(page_data, indent=2)}") # Debug log payload
        response = retry_api_call(
            requests.post,
            url,
            headers=headers,
            json=page_data
        )
        response.raise_for_status() # Check for HTTP errors
        logger.info(f"{Colors.GREEN}Successfully added job to Notion: {company} - {job_title}{Colors.RESET}")
        return True
    except requests.exceptions.RequestException as e:
         # More robust error handling
         status_code = getattr(e.response, 'status_code', 'N/A') if hasattr(e, 'response') else 'N/A'
         error_content = getattr(e.response, 'text', 'N/A') if hasattr(e, 'response') else str(e)
         logger.error(f"{Colors.RED}Failed to add job '{company} - {job_title}' to Notion. Status: {status_code}. Error: {e}. Details: {error_content[:500]}{Colors.RESET}")
         # Log the data that failed for debugging
         logger.debug(f"Failed payload data: {json.dumps(page_data, indent=2)}")
         return False
    except Exception as e: # Catch other potential errors
        logger.error(f"{Colors.RED}Unexpected error adding job '{company} - {job_title}' to Notion: {e}{Colors.RESET}", exc_info=True)
        logger.debug(f"Failed payload data during unexpected error: {json.dumps(page_data, indent=2)}")
        return False

def update_notion_table(jobs, existing_jobs):
    """Add new job postings to Notion database, avoiding duplicates"""
    if not jobs:
        logger.info("No jobs provided to update Notion.")
        return 0, 0

    added_count = 0
    skipped_count = 0

    logger.info(f"Comparing {len(jobs)} extracted jobs against {len(existing_jobs)} existing Notion entries.")

    for job in jobs:
        job_id = job.get("job_id") # Assumes job_id was added during extraction

        if not job_id:
            logger.warning(f"Skipping job due to missing job_id: {job.get('job_title', 'N/A')}")
            skipped_count += 1
            continue

        if job_id in existing_jobs:
            # logger.debug(f"Skipping duplicate job: {job.get('job_title', 'N/A')} (ID: {job_id})")
            skipped_count += 1
            continue

        # Add job to Notion
        if add_job_to_notion(job):
            added_count += 1
            existing_jobs[job_id] = True # Add to local cache to prevent re-adding in same run
            time.sleep(0.5) # Add a small delay between Notion API writes (good practice)
        else:
             logger.warning(f"Failed to add job {job.get('job_title', 'N/A')} to Notion.")
             # Optional: Decide if you want to count this as skipped or handle differently

    logger.info(f"{Colors.CYAN}Notion update summary: Added={added_count}, Skipped(duplicates/errors)={skipped_count + (len(jobs) - added_count - skipped_count)}.{Colors.RESET}")
    return added_count, skipped_count


# --- Main Execution ---

def parse_arguments():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description='Automate job hunting using AI and Notion.')
    parser.add_argument('--url', required=True, help='Target entry website URL (e.g., company career page, job board)')
    parser.add_argument('--objective', required=True, help='Job search objective (e.g., "Remote Python Developer")')
    parser.add_argument('--max-links', type=int, default=MAX_URLS_TO_PROCESS, help=f'Maximum number of relevant links to process (default: {MAX_URLS_TO_PROCESS})')
    parser.add_argument('--verbose', action='store_true', help='Enable debug logging')
    return parser.parse_args()

def main():
    """Main job hunting automation function"""
    args = parse_arguments()

    # Override config from args if provided
    global MAX_URLS_TO_PROCESS # Allow modification of global based on args
    MAX_URLS_TO_PROCESS = args.max_links

    # Set logging level
    if args.verbose:
        logger.setLevel(logging.DEBUG)
        logger.debug("Verbose logging enabled.")
    else:
         logger.setLevel(logging.INFO)


    # --- Initial Checks ---
    missing_keys = []
    if not FIRECRAWL_API_KEY: missing_keys.append("FIRECRAWL_API_KEY")
    if not GOOGLE_API_KEY: missing_keys.append("GOOGLE_API_KEY")
    if not NOTION_API_KEY: missing_keys.append("NOTION_API_KEY")
    if not NOTION_DATABASE_ID: missing_keys.append("NOTION_DATABASE_ID")

    if missing_keys:
        logger.error(f"{Colors.RED}Missing required environment variables: {', '.join(missing_keys)}{Colors.RESET}")
        logger.error(f"{Colors.YELLOW}Please ensure these are set in your .env file or environment.{Colors.RESET}")
        sys.exit(1) # Exit if essential keys are missing

    if not gemini_model:
         logger.error(f"{Colors.RED}Gemini model failed to initialize. Cannot proceed without AI capabilities.{Colors.RESET}")
         sys.exit(1)
    if not app:
        logger.error(f"{Colors.RED}FirecrawlApp failed to initialize. Cannot proceed without web scraping.{Colors.RESET}")
        sys.exit(1)

    logger.info(f"Starting job hunt for objective: '{args.objective}' on URL: {args.url}")
    logger.info(f"Will process up to {MAX_URLS_TO_PROCESS} relevant links.")

    # --- Workflow ---
    # 1. Generate Search Keyword(s)
    keyword = generate_search_parameter(args.objective)
    if not keyword:
         logger.error("Failed to generate a search keyword. Exiting.")
         sys.exit(1)

    # 2. Find potential job links from the entry URL
    candidate_links = map_website_with_keyword(args.url, keyword)
    if not candidate_links:
        logger.warning(f"{Colors.YELLOW}No potential job links found on the initial page: {args.url}. Process might yield few results.{Colors.RESET}")
        # Allow continuation, maybe direct extraction on entry URL is needed?
        # Or consider adding args.url itself to the list if it might be a direct job board
        candidate_links = [args.url] # Option: Try processing the entry URL directly
        logger.info(f"Adding entry URL {args.url} to candidates for direct processing.")


    # 3. Rank candidate URLs (if any found beyond the entry URL)
    if len(candidate_links) > 1: # Only rank if we found more than just the entry URL
         top_urls = rank_candidate_urls(candidate_links, args.objective)
    else:
         top_urls = candidate_links # Process the single URL (entry or the only one found)

    if not top_urls:
        logger.error(f"{Colors.RED}No relevant URLs to process after ranking/selection. Exiting.{Colors.RESET}")
        sys.exit(0) # Exit gracefully if nothing to process

    logger.info(f"Selected {len(top_urls)} URLs for detailed extraction: {top_urls}")

    # 4. Get Existing Jobs from Notion
    existing_jobs = get_existing_jobs_from_notion() # Fetch existing job IDs

    # 5. Extract Job Data from selected URLs
    all_job_listings = []
    logger.info(f"{Colors.CYAN}--- Starting Job Extraction from Selected URLs ---{Colors.RESET}")
    for i, link in enumerate(top_urls):
        logger.info(f"Processing URL {i+1}/{len(top_urls)}: {link}")
        extracted_jobs = extract_job_data_from_page(link, args.objective)

        if extracted_jobs is not None and isinstance(extracted_jobs, list):
            if extracted_jobs: # If list is not empty
                 logger.info(f"Found {len(extracted_jobs)} potential jobs on {link}")
                 all_job_listings.extend(extracted_jobs)
            # else: logger.info(f"No relevant jobs found by Gemini on {link}") # Already logged in function
        elif extracted_jobs is None:
             logger.warning(f"Extraction failed or returned unexpected result for URL: {link}")

        # Optional delay between processing each URL
        if i < len(top_urls) - 1: # Don't sleep after the last one
             time.sleep(1.5) # Slightly longer delay?

    logger.info(f"{Colors.CYAN}--- Job Extraction Finished ---{Colors.RESET}")

    # 6. Update Notion Table
    if all_job_listings:
        logger.info(f"{Colors.CYAN}Found total {len(all_job_listings)} potential job listings across all processed URLs.{Colors.RESET}")
        logger.info(f"{Colors.CYAN}--- Updating Notion Database ---{Colors.RESET}")
        added, skipped = update_notion_table(all_job_listings, existing_jobs)
        logger.info(f"{Colors.GREEN}--- Notion Update Complete ---{Colors.RESET}")
        final_message = f"Job search process finished. Added {added} new jobs to Notion. Skipped {skipped} duplicates/errors."
    else:
        logger.info(f"{Colors.YELLOW}No relevant job listings found matching the objective across all processed URLs.{Colors.RESET}")
        final_message = "Job search process finished. No new jobs found or added to Notion."

    logger.info(f"{Colors.GREEN}{final_message}{Colors.RESET}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info(f"\n{Colors.YELLOW}Process interrupted by user.{Colors.RESET}")
        sys.exit(0)
    except Exception as e:
        logger.error(f"{Colors.RED}An unexpected error occurred in the main process: {str(e)}{Colors.RESET}", exc_info=True) # Log traceback
        sys.exit(1)