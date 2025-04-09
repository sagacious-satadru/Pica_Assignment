import os
import json
import requests
import argparse
import time
import logging
from typing import List, Dict, Optional, Any
from dotenv import load_dotenv
from firecrawl import FirecrawlApp
import anthropic
import hashlib
import configparser
import sys

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
        # Create default config
        config['API'] = {
            'claude_model': 'claude-3-7-sonnet-20250219',
            'api_timeout': '60',
            'max_retries': '3',
            'retry_delay': '2'
        }
        config['EXTRACTION'] = {
            'max_urls_to_process': '3',
            'description_max_length': '1900'
        }
        with open(CONFIG_FILE, 'w') as f:
            config.write(f)
    else:
        config.read(CONFIG_FILE)
    
    return config

CONFIG = load_config()

# Retrieve API keys and configuration from environment variables
FIRECRAWL_API_KEY = os.getenv("FIRECRAWL_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
NOTION_API_KEY = os.getenv("NOTION_API_KEY")
NOTION_DATABASE_ID = os.getenv("NOTION_DATABASE_ID")

# API settings from config
CLAUDE_MODEL = CONFIG.get('API', 'claude_model', fallback='claude-3-7-sonnet-20250219')
API_TIMEOUT = CONFIG.getint('API', 'api_timeout', fallback=60)
MAX_RETRIES = CONFIG.getint('API', 'max_retries', fallback=3)
RETRY_DELAY = CONFIG.getint('API', 'retry_delay', fallback=2)
MAX_URLS_TO_PROCESS = CONFIG.getint('EXTRACTION', 'max_urls_to_process', fallback=3)
DESCRIPTION_MAX_LENGTH = CONFIG.getint('EXTRACTION', 'description_max_length', fallback=1900)

# Initialize FirecrawlApp and Claude client
app = FirecrawlApp(api_key=FIRECRAWL_API_KEY)
client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

def retry_api_call(func, *args, **kwargs):
    """Retry an API call with exponential backoff"""
    for attempt in range(MAX_RETRIES):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            wait_time = RETRY_DELAY * (2 ** attempt)
            if attempt < MAX_RETRIES - 1:
                logger.warning(f"API call failed: {str(e)}. Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                logger.error(f"API call failed after {MAX_RETRIES} attempts: {str(e)}")
                raise

# ---------------------------
# Step 1: Define Job-Specific Prompts
# ---------------------------

def generate_search_parameter(objective):
    """Generate a search keyword based on job objective using Claude API"""
    map_prompt = f"""
    The map function generates a list of URLs from a website and accepts a search parameter.
    Given the job objective: {objective}, provide 1-2 words that best represent a keyword search for relevant job postings.
    Only respond with the keyword(s).
    """
    logger.info(f"{Colors.YELLOW}Generating search keyword...{Colors.RESET}")
    
    try:
        completion = retry_api_call(
            client.messages.create,
            model=CLAUDE_MODEL,
            max_tokens=100,
            messages=[{"role": "user", "content": map_prompt}]
        )
        keyword = completion.content[0].text.strip()
        logger.info(f"{Colors.GREEN}Search keyword: {keyword}{Colors.RESET}")
        return keyword
    except Exception as e:
        logger.error(f"{Colors.RED}Failed to generate search parameter: {str(e)}{Colors.RESET}")
        return objective.split()[0]  # Fallback to first word of objective

def map_website_with_keyword(url, keyword):
    """Map website to find candidate job URLs using the keyword"""
    logger.info(f"{Colors.YELLOW}Mapping website {url} with search parameter: {keyword}{Colors.RESET}")
    
    try:
        map_response = retry_api_call(app.map_url, url, params={"search": keyword})
        
        # Attempt to parse the map response
        if isinstance(map_response, dict):
            links = map_response.get('urls', []) or map_response.get('links', [])
        elif isinstance(map_response, str):
            parsed = json.loads(map_response)
            links = parsed.get('urls', []) or parsed.get('links', [])
        elif isinstance(map_response, list):
            links = map_response
        else:
            links = []
            
        logger.info(f"{Colors.GREEN}Found {len(links)} candidate links.{Colors.RESET}")
        return links
    except Exception as e:
        logger.error(f"{Colors.RED}Error mapping website: {str(e)}{Colors.RESET}")
        return []

def rank_candidate_urls(links, objective):
    """Rank candidate URLs by relevance to job objective using Claude API"""
    if not links:
        return []
        
    rank_prompt = f"""
    You are given a list of URLs and a job objective: {objective}.
    Rank the URLs to choose the top {MAX_URLS_TO_PROCESS} that are most likely to contain job postings.
    Provide a JSON array with exactly {MAX_URLS_TO_PROCESS} objects where each object includes:
    - "url": the full URL,
    - "relevance_score": a number between 0 and 100 indicating relevance,
    - "reason": a short explanation.
    Return ONLY valid JSON.
    URLs to analyze:
    {json.dumps(links[:100], indent=2)}  # Limit to 100 links to avoid token limits
    """
    logger.info(f"{Colors.YELLOW}Ranking candidate URLs...{Colors.RESET}")
    
    try:
        completion = retry_api_call(
            client.messages.create,
            model=CLAUDE_MODEL,
            max_tokens=1000,
            messages=[{"role": "user", "content": rank_prompt}]
        )
        
        # Clean and parse response
        response_text = completion.content[0].text.strip()
        # Extract JSON from markdown code block if present
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0].strip()
            
        ranked_results = json.loads(response_text)
        top_urls = [entry["url"] for entry in ranked_results]
        logger.info(f"{Colors.GREEN}Top ranked URLs: {top_urls}{Colors.RESET}")
        return top_urls
    except Exception as e:
        logger.error(f"{Colors.RED}Error ranking URLs: {str(e)}{Colors.RESET}")
        # Fallback to first MAX_URLS_TO_PROCESS links if ranking fails
        return links[:MAX_URLS_TO_PROCESS] if links else []

def extract_job_data_from_page(url, objective):
    """Extract job posting data from a webpage using Claude API"""
    logger.info(f"{Colors.YELLOW}Scraping URL for job data: {url}{Colors.RESET}")
    
    try:
        scrape_result = retry_api_call(app.scrape_url, url, params={'formats': ['markdown']})
        
        # Improved prompt for better extraction
        check_prompt = f"""
        You are provided with web page content and a job objective.
        The objective is: {objective}
        
        The content of the page is given below in Markdown format.
        
        If the page contains job listings that match the objective, extract the following fields for EACH job posting:
          - "job_title": The title of the job position
          - "company": Company name offering the job
          - "location": Location of the job (in-office, remote, hybrid, or city/state)
          - "job_description": A concise summary of the job responsibilities and requirements
          - "application_url": URL to apply for the job (if available, otherwise use the page URL)
        
        FORMAT REQUIREMENTS:
        1. Return a JSON array where each object represents one distinct job posting
        2. For job_description, limit to 300-400 words maximum, focusing on key responsibilities and requirements
        3. Ensure all fields contain values (use "Unknown" if information is missing)
        4. If a field has HTML/markdown formatting, clean it to plain text
        
        If the content does not meet the objective or no job listings are found, reply exactly 'Objective not met'.
        
        Page content:
        {scrape_result.get('markdown', '')}
        """
        
        completion = retry_api_call(
            client.messages.create,
            model=CLAUDE_MODEL,
            max_tokens=1500,
            messages=[{"role": "user", "content": check_prompt}]
        )
        
        result_text = completion.content[0].text.strip()
        # Extract JSON from markdown code block if present
        if "```json" in result_text:
            result_text = result_text.split("```json")[1].split("```")[0].strip()
        elif "```" in result_text:
            result_text = result_text.split("```")[1].split("```")[0].strip()
        else:
            result_text = result_text.strip()
            
        if result_text == "Objective not met":
            logger.info(f"{Colors.YELLOW}Objective not met for URL: {url}{Colors.RESET}")
            return None
            
        job_data = json.loads(result_text)
        
        # Process job descriptions to ensure they're not too long for Notion
        for job in job_data:
            if len(job.get("job_description", "")) > DESCRIPTION_MAX_LENGTH:
                job["job_description"] = job["job_description"][:DESCRIPTION_MAX_LENGTH] + "..."
            
            # Set a fallback application URL if missing
            if not job.get("application_url"):
                job["application_url"] = url
                
            # Add a job_id for deduplication
            job_string = f"{job.get('job_title', '')}-{job.get('company', '')}-{job.get('location', '')}"
            job["job_id"] = hashlib.md5(job_string.encode()).hexdigest()
                
        logger.info(f"{Colors.GREEN}Extracted {len(job_data)} job listings from URL: {url}{Colors.RESET}")
        return job_data
    except Exception as e:
        logger.error(f"{Colors.RED}Error extracting job data: {str(e)}{Colors.RESET}")
        return None

# ---------------------------
# Step 2: Notion Integration
# ---------------------------

def get_existing_jobs_from_notion():
    """Retrieve existing jobs from Notion database to avoid duplicates"""
    url = f"https://api.notion.com/v1/databases/{NOTION_DATABASE_ID}/query"
    headers = {
        "Authorization": f"Bearer {NOTION_API_KEY}",
        "Content-Type": "application/json",
        "Notion-Version": "2022-06-28"
    }
    
    existing_jobs = {}
    has_more = True
    start_cursor = None
    
    try:
        while has_more:
            data = {}
            if start_cursor:
                data["start_cursor"] = start_cursor
                
            response = retry_api_call(
                requests.post,
                url, 
                headers=headers,
                json=data,
                timeout=API_TIMEOUT
            )
            
            response.raise_for_status()
            result = response.json()
            
            for page in result["results"]:
                props = page["properties"]
                job_title = props.get("Job Title", {}).get("title", [{}])[0].get("text", {}).get("content", "")
                company = props.get("Company", {}).get("rich_text", [{}])[0].get("text", {}).get("content", "")
                location = props.get("Location", {}).get("rich_text", [{}])[0].get("text", {}).get("content", "")
                
                job_string = f"{job_title}-{company}-{location}"
                job_id = hashlib.md5(job_string.encode()).hexdigest()
                existing_jobs[job_id] = True
                
            has_more = result.get("has_more", False)
            start_cursor = result.get("next_cursor")
            
        logger.info(f"{Colors.CYAN}Found {len(existing_jobs)} existing jobs in Notion{Colors.RESET}")
        return existing_jobs
    except Exception as e:
        logger.error(f"{Colors.RED}Error retrieving existing jobs: {str(e)}{Colors.RESET}")
        return {}

def add_job_to_notion(job):
    """Add a single job posting to Notion database"""
    url = "https://api.notion.com/v1/pages"
    headers = {
        "Authorization": f"Bearer {NOTION_API_KEY}",
        "Content-Type": "application/json",
        "Notion-Version": "2022-06-28"
    }
    
    # Ensure job description is not too long (Notion has a 2000 char limit for rich_text)
    job_description = job.get("job_description", "N/A")
    if len(job_description) > DESCRIPTION_MAX_LENGTH:
        job_description = job_description[:DESCRIPTION_MAX_LENGTH] + "..."
    
    data = {
        "parent": {"database_id": NOTION_DATABASE_ID},
        "properties": {
            "Job Title": {
                "title": [{"text": {"content": job.get("job_title", "N/A")}}]
            },
            "Company": {
                "rich_text": [{"text": {"content": job.get("company", "N/A")}}]
            },
            "Location": {
                "rich_text": [{"text": {"content": job.get("location", "N/A")}}]
            },
            "Description": {
                "rich_text": [{"text": {"content": job_description}}]
            },
            "URL": {
                "url": job.get("application_url", "")
            }
        }
    }
    
    try:
        response = retry_api_call(
            requests.post,
            url, 
            headers=headers, 
            json=data,
            timeout=API_TIMEOUT
        )
        
        response.raise_for_status()
        logger.info(f"{Colors.GREEN}Successfully added job: {job.get('job_title', 'N/A')}{Colors.RESET}")
        return True
    except Exception as e:
        logger.error(f"{Colors.RED}Failed to add job: {job.get('job_title', 'N/A')} - {str(e)}{Colors.RESET}")
        return False

def update_notion_table(jobs, existing_jobs=None):
    """Add new job postings to Notion database, avoiding duplicates"""
    if existing_jobs is None:
        existing_jobs = get_existing_jobs_from_notion()
        
    added_count = 0
    skipped_count = 0
    
    for job in jobs:
        job_id = job.get("job_id")
        
        if job_id in existing_jobs:
            logger.info(f"{Colors.YELLOW}Skipping duplicate job: {job.get('job_title', 'N/A')}{Colors.RESET}")
            skipped_count += 1
            continue
            
        if add_job_to_notion(job):
            added_count += 1
            existing_jobs[job_id] = True
    
    logger.info(f"{Colors.CYAN}Added {added_count} new jobs. Skipped {skipped_count} duplicates.{Colors.RESET}")
    return added_count, skipped_count

# ---------------------------
# Step 3: Main Job Hunt Automation
# ---------------------------

def parse_arguments():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description='Automate job hunting using AI')
    parser.add_argument('--url', help='Target job board URL')
    parser.add_argument('--objective', help='Job search objective (e.g., "Remote Python Developer")')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    return parser.parse_args()

def main():
    """Main job hunting automation function"""
    args = parse_arguments()
    
    # Set logging level based on verbose flag
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    # Check if API keys are available
    missing_keys = []
    if not FIRECRAWL_API_KEY:
        missing_keys.append("FIRECRAWL_API_KEY")
    if not ANTHROPIC_API_KEY:
        missing_keys.append("ANTHROPIC_API_KEY")
    if not NOTION_API_KEY:
        missing_keys.append("NOTION_API_KEY")
    if not NOTION_DATABASE_ID:
        missing_keys.append("NOTION_DATABASE_ID")
        
    if missing_keys:
        logger.error(f"{Colors.RED}Missing required API keys: {', '.join(missing_keys)}{Colors.RESET}")
        logger.error(f"{Colors.YELLOW}Please add these to your .env file and try again.{Colors.RESET}")
        return
    
    # Get job search parameters
    url = args.url or input(f"{Colors.BLUE}Enter the website URL to crawl for jobs: {Colors.RESET}")
    objective = args.objective or input(f"{Colors.BLUE}Enter your job search objective (e.g., 'Remote Python Developer'): {Colors.RESET}")
    
    # Generate search keyword
    keyword = generate_search_parameter(objective)
    
    # Map the website to get candidate URLs using that keyword
    candidate_links = map_website_with_keyword(url, keyword)
    
    if not candidate_links:
        logger.error(f"{Colors.RED}No candidate links were found. Try a different website or refine your objective.{Colors.RESET}")
        return
    
    # Rank the candidate URLs to select the most promising pages
    top_urls = rank_candidate_urls(candidate_links, objective)
    if not top_urls:
        logger.error(f"{Colors.RED}No top URLs could be ranked. Aborting process.{Colors.RESET}")
        return
    
    # Get existing jobs to avoid duplicates
    existing_jobs = get_existing_jobs_from_notion()
    
    # Extract job data from the top-ranked pages
    all_job_listings = []
    for link in top_urls:
        job_data = extract_job_data_from_page(link, objective)
        if job_data and isinstance(job_data, list):
            all_job_listings.extend(job_data)
    
    if not all_job_listings:
        logger.error(f"{Colors.RED}No job postings were extracted from the provided pages.{Colors.RESET}")
        return
    
    logger.info(f"{Colors.CYAN}Total job postings extracted: {len(all_job_listings)}{Colors.RESET}")
    
    # Update Notion with each job posting
    added, skipped = update_notion_table(all_job_listings, existing_jobs)
    
    # Final summary
    logger.info(f"{Colors.GREEN}Job search complete! Added {added} new jobs, skipped {skipped} duplicates.{Colors.RESET}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info(f"{Colors.YELLOW}Process interrupted by user.{Colors.RESET}")
        sys.exit(0)
    except Exception as e:
        logger.error(f"{Colors.RED}Unexpected error: {str(e)}{Colors.RESET}")
        sys.exit(1)
