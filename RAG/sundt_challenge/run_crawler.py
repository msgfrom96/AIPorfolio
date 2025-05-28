# Phase 1: Data Ingestion & Storage

# Base URL for the Sundt website
BASE_URL = "https://www.sundt.com"

import re
from urllib.parse import urljoin, urlparse
from typing import Any, Dict, List, Optional, Pattern, Set
import requests
from bs4 import BeautifulSoup
import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import TimeoutException, NoSuchElementException, ElementClickInterceptedException
from selenium.webdriver.common.action_chains import ActionChains
import logging
import json
import os
from datetime import datetime

# Configure logging for the crawler
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def setup_driver() -> webdriver.Chrome:
    """
    Setup Chrome driver with appropriate options for web scraping.

    Returns:
        webdriver.Chrome: Configured Chrome WebDriver instance
    """
    chrome_options = Options()
    # Run in headless mode to avoid opening browser window
    chrome_options.add_argument("--headless")
    # Security and stability options for containerized environments
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--disable-gpu")
    # Set window size for consistent rendering
    chrome_options.add_argument("--window-size=1920,1080")
    # Use realistic user agent to avoid bot detection
    chrome_options.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36")

    driver = webdriver.Chrome(options=chrome_options)
    return driver

def fetch_html_static(url: str, session: requests.Session) -> Optional[str]:
    """
    Fetch HTML content using requests for static pages.

    Args:
        url: The URL to fetch
        session: Requests session for connection pooling

    Returns:
        HTML content as string or None if failed
    """
    try:
        resp = session.get(url, headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}, timeout=10)
        resp.raise_for_status()
        return resp.text
    except requests.RequestException as e:
        logger.error(f"Error fetching {url}: {e}")
        return None

def wait_for_load_more_button(driver: webdriver.Chrome, timeout: int = 10) -> Optional[webdriver.remote.webelement.WebElement]:
    """
    Wait for and find the load more button using multiple selector strategies.

    Args:
        driver: Chrome WebDriver instance
        timeout: Maximum time to wait for button

    Returns:
        WebElement if found, None otherwise
    """
    wait = WebDriverWait(driver, timeout)

    # Multiple selectors to handle different button implementations
    selectors = [
        "//button[contains(@class, 'facetwp-load-more')]",
        "//button[contains(text(), 'Load More')]",
        "//button[contains(text(), 'Show More')]",
        "//a[contains(text(), 'Load More')]",
        "//a[contains(text(), 'Show More')]",
        "//button[contains(@class, 'load-more')]"
    ]

    # Try each selector until one works
    for selector in selectors:
        try:
            element = wait.until(EC.element_to_be_clickable((By.XPATH, selector)))
            return element
        except TimeoutException:
            continue

    return None

def click_load_more_with_retry(driver: webdriver.Chrome, button: webdriver.remote.webelement.WebElement, max_retries: int = 3) -> bool:
    """
    Attempt to click the load more button with multiple strategies and retry logic.

    Args:
        driver: Chrome WebDriver instance
        button: The button element to click
        max_retries: Maximum number of click attempts

    Returns:
        True if click was successful, False otherwise
    """

    for attempt in range(max_retries):
        try:
            # Strategy 1: Scroll to element and perform regular click
            driver.execute_script("arguments[0].scrollIntoView({behavior: 'smooth', block: 'center'});", button)
            time.sleep(1)

            # Wait for any overlays to disappear
            WebDriverWait(driver, 5).until(EC.element_to_be_clickable(button))

            # Try regular click first
            button.click()
            return True

        except ElementClickInterceptedException:
            try:
                # Strategy 2: Use JavaScript click to bypass overlays
                driver.execute_script("arguments[0].click();", button)
                return True
            except Exception:
                try:
                    # Strategy 3: Use ActionChains for more precise clicking
                    actions = ActionChains(driver)
                    actions.move_to_element(button).click().perform()
                    return True
                except Exception:
                    if attempt < max_retries - 1:
                        logger.warning(f"Click attempt {attempt + 1} failed, retrying...")
                        time.sleep(2)
                    continue
        except Exception as e:
            if attempt < max_retries - 1:
                logger.warning(f"Click attempt {attempt + 1} failed with error: {e}, retrying...")
                time.sleep(2)
            continue

    return False

def get_current_project_count(driver: webdriver.Chrome) -> int:
    """
    Get the current number of projects loaded on the page by counting project links.

    Args:
        driver: Chrome WebDriver instance

    Returns:
        Number of project links found
    """
    try:
        # Parse current page source to count project links
        soup = BeautifulSoup(driver.page_source, 'html.parser')
        project_links = soup.find_all('a', href=re.compile(r'/projects/[^/]+/?$'))
        return len(project_links)
    except Exception:
        return 0

def parse_projects_data(project_soup: BeautifulSoup, project_url: str) -> Dict[str, Any]:
    """
    Parse detailed project data from individual project page HTML.

    Args:
        project_soup: BeautifulSoup object of project page
        project_url: URL of the project page

    Returns:
        Dictionary containing structured project data
    """

    # Extract project name from the main title element
    project_name = ""
    title_elem = project_soup.select_one('.entry-title')
    if title_elem:
        project_name = title_elem.get_text(strip=True)

    # Extract metadata from structured list items (location, client, etc.)
    metadata = {}
    list_info_items = project_soup.select('.list-info li')
    for item in list_info_items:
        text = item.get_text(strip=True)
        if ':' in text:
            key, value = text.split(':', 1)
            metadata[key.strip()] = value.strip()

    # Extract image URLs from gallery links
    image_urls = []
    gallery_links = project_soup.select('.slider__slides a.mfp-gallery')
    for link in gallery_links:
        href = link.get('href')
        if href:
            image_urls.append(urljoin(project_url, href))

    # Extract project features from bullet point lists
    features = []
    bullet_items = project_soup.select('.list-bullets li')
    for item in bullet_items:
        feature_text = item.get_text(strip=True)
        if feature_text:
            features.append(feature_text)

    # Extract project overview from description paragraphs
    overview = ""
    overview_paragraphs = project_soup.select('.section--light-gray .section__content p')
    overview_texts = []
    for p in overview_paragraphs:
        p_text = p.get_text(strip=True)
        if p_text:
            overview_texts.append(p_text)
    overview = ' '.join(overview_texts)

    # Extract similar/related projects with multiple parsing strategies
    similar_projects = []

    # Strategy 1: Look for carousel items with direct links
    carousel_links = project_soup.select('.projects-carousel a.component')
    for link in carousel_links:
        href = link.get('href')
        if href:
            # Get the title from various possible locations
            title = ""

            # Try to get title from title attribute first (hover text)
            title_attr = link.get('title')
            if title_attr:
                title = title_attr.strip()
            else:
                # Try to get from text content
                title = link.get_text(strip=True)

            # If still no title, try to get from nested elements
            if not title:
                # Look for h3, h4, or other heading elements
                heading = link.find(['h3', 'h4', 'h5', 'h6'])
                if heading:
                    title = heading.get_text(strip=True)
                else:
                    # Look for any text in spans or divs
                    text_elem = link.find(['span', 'div'])
                    if text_elem:
                        title = text_elem.get_text(strip=True)

            if title and href:
                similar_projects.append({
                    'title': title,
                    'link': urljoin(project_url, href)
                })

    # Strategy 2: Look for alternative carousel structures if first strategy failed
    if not similar_projects:
        # Try different selectors for similar projects
        alternative_selectors = [
            '.projects-carousel .component',
            '.related-projects a',
            '.similar-projects a',
            '.project-carousel a',
            '[class*="carousel"] a[href*="/projects/"]'
        ]

        for selector in alternative_selectors:
            carousel_items = project_soup.select(selector)
            for item in carousel_items:
                href = item.get('href')
                if href and '/projects/' in href:
                    title = ""

                    # Check for title attribute
                    title_attr = item.get('title')
                    if title_attr:
                        title = title_attr.strip()
                    else:
                        # Get text content
                        title = item.get_text(strip=True)

                    # Look for nested elements with project names
                    if not title:
                        for tag in ['h3', 'h4', 'h5', 'h6', 'span', 'div', 'p']:
                            elem = item.find(tag)
                            if elem:
                                text = elem.get_text(strip=True)
                                if text and len(text) > 3:  # Avoid empty or very short text
                                    title = text
                                    break

                    if title and href:
                        similar_projects.append({
                            'title': title,
                            'link': urljoin(project_url, href)
                        })

            # If we found projects with this selector, break
            if similar_projects:
                break

    # Strategy 3: Look for data attributes or JavaScript-populated content
    if not similar_projects:
        # Look for elements with data attributes that might contain project info
        data_elements = project_soup.find_all(attrs={'data-title': True})
        for elem in data_elements:
            data_title = elem.get('data-title')
            href = elem.get('href') or elem.find('a', href=True)

            if data_title and href:
                if isinstance(href, str):
                    link_url = href
                else:
                    link_url = href.get('href')

                if link_url and '/projects/' in link_url:
                    similar_projects.append({
                        'title': data_title.strip(),
                        'link': urljoin(project_url, link_url)
                    })

    # Remove duplicates based on URL to avoid redundant entries
    seen_urls = set()
    unique_similar_projects = []
    for project in similar_projects:
        if project['link'] not in seen_urls:
            seen_urls.add(project['link'])
            unique_similar_projects.append(project)

    return {
        'project_name': project_name,
        'metadata': metadata,
        'image_urls': image_urls,
        'features': features,
        'overview': overview,
        'similar_projects': unique_similar_projects,
        'project_url': project_url
    }

def parse_awards_data(award_soup: BeautifulSoup, award_url: str) -> List[Dict[str, Any]]:
    """
    Parse detailed awards data from awards page with comprehensive year and location extraction.

    Args:
        award_soup: BeautifulSoup object of awards page
        award_url: URL of the awards page

    Returns:
        List of dictionaries containing structured award data
    """
    awards = []

    # Strategy 1: Find awards in the main content sections
    # Look for awards in the two-column section (.content-two-column)
    award_sections = award_soup.find_all('div', class_='col')

    for section in award_sections:
        # Find all h6 elements with class "text--serif" that contain award names
        award_headers = section.find_all('h6', class_='text--serif')

        for header in award_headers:
            # Extract title from the em tag within h6
            title_elem = header.find('em')
            if not title_elem:
                continue

            title = title_elem.get_text(strip=True)

            # Initialize award data fields
            awarded_by = ""
            project_name = ""
            project_link = ""
            category = ""
            location = ""
            year = ""

            # Find the next sibling elements to get award details
            current_elem = header.next_sibling
            award_details = []
            all_text_content = ""  # Collect all text for comprehensive parsing

            # Collect the next few p elements that contain award details
            # Look at next 5 elements to capture all details, stopping if we hit another h6
            for _ in range(5):
                while current_elem and current_elem.name != 'p' and (current_elem.name != 'h6' or 'text--serif' not in current_elem.get('class', [])):
                    current_elem = current_elem.next_sibling
                # Stop if we hit the next award header
                if current_elem and current_elem.name == 'h6' and 'text--serif' in current_elem.get('class', []):
                    break

                if current_elem and current_elem.name == 'p':
                    p_text = current_elem.get_text(strip=True)
                    if p_text:  # Only add non-empty paragraphs
                        award_details.append(p_text)
                        all_text_content += " " + p_text  # Accumulate all text

                        # Look for project links within this p element
                        project_link_elem = current_elem.find('a', href=re.compile(r'/projects/'))
                        if project_link_elem:
                            project_link = urljoin(award_url, project_link_elem['href'])
                            # Prefer text from the link if available, otherwise try to extract later
                            if not project_name:
                                project_name = project_link_elem.get_text(strip=True)

                        # Extract awarded_by (usually in strong tags or first line)
                        org_elem = current_elem.find('strong')
                        if org_elem and not awarded_by:
                            awarded_by = org_elem.get_text(strip=True)
                        elif not awarded_by and award_details:
                             # If no strong tag, first detail might be the organization
                             # Check if the first detail line looks like an organization name (doesn't contain year or common project/category terms)
                             first_line = award_details[0]
                             if not re.search(r'\b\d{4}\b', first_line) and 'Category' not in first_line and '/projects/' not in first_line:
                                 awarded_by = first_line

                        current_elem = current_elem.next_sibling
                    else:
                        current_elem = current_elem.next_sibling
                else:
                    break

            # Combine award details into a single string for easier parsing
            raw_details_string = ' | '.join(award_details)

            # --- Improved Location and Year Extraction ---
            # Iterate through segments separated by '|' in raw_details_string
            segments = [s.strip() for s in raw_details_string.split('|')]

            # Regex for City, State pattern, trying to be more specific
            # Look for patterns like "City, ST" or "City Name, ST"
            # Prioritize patterns followed by a year or at the end of a segment
            location_year_pattern = re.compile(r'([A-Z][a-z]+(?: [A-Z][a-z]+)?,\s*[A-Z]{2})\s*(\d{4})?')
            location_only_pattern = re.compile(r'([A-Z][a-z]+(?: [A-Z][a-z]+)?,\s*[A-Z]{2})\s*$')
            year_pattern = re.compile(r'\b(\d{4})\b') # Simple year pattern

            found_years = []
            found_location = ""

            for segment in segments:
                # Try to find Location followed by Year first
                match = location_year_pattern.search(segment)
                if match:
                    found_location = match.group(1)
                    if match.group(2):
                        found_years.append(int(match.group(2)))
                    # Once a location is found, we can stop searching for location in other segments
                    # But continue searching for years in all segments

                # If location not found yet, try Location at the end of the segment
                if not found_location:
                    match = location_only_pattern.search(segment)
                    if match:
                        found_location = match.group(1)

                # Find all years in the segment
                year_matches = year_pattern.findall(segment)
                found_years.extend([int(y) for y in year_matches if 1980 <= int(y) <= 2030]) # Filter for reasonable years

            # Assign extracted location and year
            location = found_location if found_location else ""

            # Take the most recent valid year found across all segments
            if found_years:
                 year = str(max(found_years))
            else:
                 year = "" # Ensure year is empty string if none found

            # --- End Improved Extraction ---


            # Post-process to clean up data
            # If project_name is empty but we have award details, try to extract from details
            # Look for text that is not an organization, category, location, or year
            if not project_name and award_details:
                for detail in award_details:
                    # Check if it looks like a project name (longer, doesn't contain common award keywords)
                    if len(detail) > 15 and not re.search(r'\b\d{4}\b', detail) and 'Category' not in detail and not re.search(r'[A-Z][a-z]+,\s*[A-Z]{2}', detail):
                         # Check if it's not an organization name (simple check)
                         org_keywords = ['Engineering', 'News-Record', 'Associated', 'General', 'Contractors', 'America', 'Chapter', 'Association']
                         if not any(keyword in detail for keyword in org_keywords):
                             project_name = detail.strip()
                             break # Found a potential project name, stop searching

            # Clean up awarded_by to remove extra text
            if awarded_by:
                # Remove common suffixes and leading/trailing whitespace
                awarded_by = re.sub(r'\s*(Chapter|Division|Association).*$', '', awarded_by, flags=re.IGNORECASE)
                awarded_by = awarded_by.strip()

            # Extract category if not found yet, looking for "Category" keyword
            if not category and award_details:
                 for detail in award_details:
                     category_match = re.search(r'(.*Category)', detail, re.IGNORECASE)
                     if category_match:
                         category = category_match.group(1).strip()
                         break

            # Only add award if we have meaningful data (title is required, plus at least one other key piece)
            if title and (awarded_by or project_name or year or location):
                awards.append({
                    'title': title,
                    'awarded_by': awarded_by,
                    'project_name': project_name,
                    'project_link': project_link,
                    'category': category,
                    'location': location,
                    'year': year,
                    'raw_details': raw_details_string,  # Keep raw details for debugging
                    'all_text': all_text_content.strip()  # Keep all text for debugging
                })

    # Strategy 2: Look for awards in hidden sections that might be revealed by "Load More"
    # Note: The awards page doesn't seem to use dynamic loading based on observation,
    # but keeping this structure allows for robustness if the site changes.
    # The parsing logic is the same as Strategy 1.
    hidden_sections = award_soup.find_all('div', class_='hidden')
    for hidden_section in hidden_sections:
        # Apply the same parsing logic to hidden content
        award_headers = hidden_section.find_all('h6', class_='text--serif')

        for header in award_headers:
            title_elem = header.find('em')
            if not title_elem:
                continue

            title = title_elem.get_text(strip=True)

            # Similar parsing logic as above for hidden sections
            awarded_by = ""
            project_name = ""
            project_link = ""
            category = ""
            location = ""
            year = ""

            current_elem = header.next_sibling
            award_details = []
            all_text_content = ""

            for _ in range(5):
                while current_elem and current_elem.name != 'p' and (current_elem.name != 'h6' or 'text--serif' not in current_elem.get('class', [])):
                    current_elem = current_elem.next_sibling
                if current_elem and current_elem.name == 'h6' and 'text--serif' in current_elem.get('class', []):
                    break

                if current_elem and current_elem.name == 'p':
                    p_text = current_elem.get_text(strip=True)
                    if p_text:
                        award_details.append(p_text)
                        all_text_content += " " + p_text

                        project_link_elem = current_elem.find('a', href=re.compile(r'/projects/'))
                        if project_link_elem:
                            project_link = urljoin(award_url, project_link_elem['href'])
                            if not project_name:
                                project_name = project_link_elem.get_text(strip=True)

                        org_elem = current_elem.find('strong')
                        if org_elem and not awarded_by:
                            awarded_by = org_elem.get_text(strip=True)
                        elif not awarded_by and award_details:
                             first_line = award_details[0]
                             if not re.search(r'\b\d{4}\b', first_line) and 'Category' not in first_line and '/projects/' not in first_line:
                                 awarded_by = first_line

                        current_elem = current_elem.next_sibling
                    else:
                        current_elem = current_elem.next_sibling
                else:
                    break

            raw_details_string = ' | '.join(award_details)
            segments = [s.strip() for s in raw_details_string.split('|')]

            found_years = []
            found_location = ""

            for segment in segments:
                match = location_year_pattern.search(segment)
                if match:
                    found_location = match.group(1)
                    if match.group(2):
                        found_years.append(int(match.group(2)))

                if not found_location:
                    match = location_only_pattern.search(segment)
                    if match:
                        found_location = match.group(1)

                year_matches = year_pattern.findall(segment)
                found_years.extend([int(y) for y in year_matches if 1980 <= int(y) <= 2030])

            location = found_location if found_location else ""
            if found_years:
                 year = str(max(found_years))
            else:
                 year = ""

            if not project_name and award_details:
                for detail in award_details:
                    if len(detail) > 15 and not re.search(r'\b\d{4}\b', detail) and 'Category' not in detail and not re.search(r'[A-Z][a-z]+,\s*[A-Z]{2}', detail):
                         org_keywords = ['Engineering', 'News-Record', 'Associated', 'General', 'Contractors', 'America', 'Chapter', 'Association']
                         if not any(keyword in detail for keyword in org_keywords):
                             project_name = detail.strip()
                             break

            if awarded_by:
                awarded_by = re.sub(r'\s*(Chapter|Division|Association).*$', '', awarded_by, flags=re.IGNORECASE)
                awarded_by = awarded_by.strip()

            if not category and award_details:
                 for detail in award_details:
                     category_match = re.search(r'(.*Category)', detail, re.IGNORECASE)
                     if category_match:
                         category = category_match.group(1).strip()
                         break

            if title and (awarded_by or project_name or year or location):
                awards.append({
                    'title': title,
                    'awarded_by': awarded_by,
                    'project_name': project_name,
                    'project_link': project_link,
                    'category': category,
                    'location': location,
                    'year': year,
                    'raw_details': raw_details_string,
                    'all_text': all_text_content.strip()
                })

    return awards

def get_projects_from_listing_dynamic(driver: webdriver.Chrome) -> List[Dict[str, Any]]:
    """
    Get projects from the main projects listing page using Selenium to handle dynamic content loading.

    Args:
        driver: Chrome WebDriver instance

    Returns:
        List of dictionaries containing project data
    """
    # Use urljoin with BASE_URL and the endpoint
    projects_url = urljoin(BASE_URL, "/projects/")

    try:
        driver.get(projects_url)
        wait = WebDriverWait(driver, 15)

        # Wait for initial content to load
        wait.until(EC.presence_of_element_located((By.TAG_NAME, "body")))
        time.sleep(3)

        projects = []
        load_attempts = 0
        consecutive_failures = 0
        max_consecutive_failures = 3
        previous_count = 0

        logger.info("Starting dynamic loading of projects...")

        # Continue loading more content until no more is available
        while consecutive_failures < max_consecutive_failures:
            # Get current project count to track progress
            current_count = get_current_project_count(driver)
            logger.info(f"Current projects loaded: {current_count}")

            # Check if we've loaded new content
            if current_count > previous_count:
                previous_count = current_count
                consecutive_failures = 0  # Reset failure counter

            # Look for load more button
            load_more_button = wait_for_load_more_button(driver, timeout=5)

            if not load_more_button:
                logger.info("No load more button found - all content may be loaded")
                break

            # Check if button is disabled or loading
            button_text = load_more_button.text.lower()
            button_class = load_more_button.get_attribute('class') or ''

            if 'loading' in button_text or 'disabled' in button_class:
                logger.info("Button is loading or disabled, waiting...")
                time.sleep(3)
                consecutive_failures += 1
                continue

            # Attempt to click the button
            success = click_load_more_with_retry(driver, load_more_button)

            if success:
                load_attempts += 1
                logger.info(f"Successfully clicked 'Load More' button (attempt {load_attempts})")

                # Wait for new content to load with progress checking
                loading_start = time.time()
                max_loading_time = 15

                while time.time() - loading_start < max_loading_time:
                    time.sleep(2)
                    new_count = get_current_project_count(driver)

                    if new_count > current_count:
                        logger.info(f"New content loaded: {new_count - current_count} additional projects")
                        break

                    # Check if button is still loading
                    try:
                        current_button = wait_for_load_more_button(driver, timeout=1)
                        if current_button:
                            button_text = current_button.text.lower()
                            if 'loading' not in button_text:
                                break
                    except:
                        break

                consecutive_failures = 0

            else:
                logger.warning("Failed to click load more button")
                consecutive_failures += 1
                time.sleep(2)

        logger.info(f"Finished loading. Total load attempts: {load_attempts}")

        # Now extract all project links from the fully loaded page
        soup = BeautifulSoup(driver.page_source, 'html.parser')
        project_links = soup.find_all('a', href=re.compile(r'/projects/[^/]+/?$'))

        logger.info(f"Found {len(project_links)} project links after dynamic loading")

        # Create a session for fetching individual project details
        session = requests.Session()

        # Process projects with better error handling and progress tracking
        processed_urls = set()

        for i, link in enumerate(project_links):
            # Use urljoin with BASE_URL and the relative link href
            project_url = urljoin(BASE_URL, link['href'])

            # Skip duplicates to avoid processing same project twice
            if project_url in processed_urls:
                continue
            processed_urls.add(project_url)

            # Get detailed project info using static requests for individual pages
            project_html = fetch_html_static(project_url, session)
            if project_html:
                project_soup = BeautifulSoup(project_html, 'html.parser')

                # Parse detailed project data
                project_data = parse_projects_data(project_soup, project_url)
                projects.append(project_data)

                if (len(projects)) % 10 == 0:
                    logger.info(f"Processed {len(projects)} projects...")

                # Add small delay to be respectful to the server
                if i % 5 == 0:
                    time.sleep(0.5)

        logger.info(f"Successfully processed {len(projects)} unique projects")
        return projects

    except Exception as e:
        logger.error(f"Error in dynamic project scraping: {e}")
        return []

def get_awards_from_page(session: requests.Session) -> List[Dict[str, Any]]:
    """
    Get awards from the awards page using static scraping with improved parsing.

    Args:
        session: Requests session for connection pooling

    Returns:
        List of dictionaries containing award data
    """
    # Use urljoin with BASE_URL and the endpoint
    awards_url = urljoin(BASE_URL, "/awards/")
    html = fetch_html_static(awards_url, session)
    if not html:
        return []

    soup = BeautifulSoup(html, 'html.parser')
    awards = parse_awards_data(soup, awards_url)

    logger.info(f"Processed {len(awards)} awards")

    # Debug: Log year extraction results for monitoring data quality
    awards_with_years = [a for a in awards if a['year']]
    awards_without_years = [a for a in awards if not a['year']]

    logger.info(f"Awards with years: {len(awards_with_years)}")
    logger.info(f"Awards without years: {len(awards_without_years)}")

    if awards_without_years:
        logger.debug("Sample awards without years (for debugging):")
        for i, award in enumerate(awards_without_years[:3]):
            logger.debug(f"  {i+1}. {award['title']}")
            logger.debug(f"     Raw details: {award['raw_details'][:100]}...")
            logger.debug(f"     All text: {award['all_text'][:100]}...")

    return awards

def crawl_site() -> Dict[str, List[Dict[str, Any]]]:
    """
    Main crawler function that orchestrates the scraping of Sundt website for projects and awards.
    Uses Selenium for dynamic content and requests for static content.

    Returns:
        Dictionary containing 'projects' and 'awards' lists
    """
    logger.info("Starting to crawl Sundt website with improved dynamic loading support...")

    # Setup Selenium driver for dynamic content
    driver = setup_driver()

    try:
        # Get projects using dynamic scraping (projects page uses infinite scroll)
        logger.info("Fetching projects with enhanced dynamic loading...")
        projects = get_projects_from_listing_dynamic(driver)

    finally:
        # Always close the driver to free resources
        driver.quit()

    # Get awards using static scraping (awards page doesn't use dynamic loading)
    logger.info("Fetching awards with improved year extraction...")
    session = requests.Session()
    awards = get_awards_from_page(session)

    return {
        'projects': projects,
        'awards': awards
    }

# Save the crawled data to JSON files for persistence and future use


def save_data_to_json(results, output_dir="data"):
    """
    Save the crawled projects and awards data to JSON files with timestamps.

    Args:
        results (dict): Dictionary containing 'projects' and 'awards' lists
        output_dir (str): Directory to save the JSON files

    Returns:
        Dictionary with paths to saved files
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Add timestamp to filenames for versioning
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save projects data separately for easier access
    projects_file = os.path.join(output_dir, f"sundt_projects_{timestamp}.json")
    with open(projects_file, 'w', encoding='utf-8') as f:
        json.dump(results['projects'], f, indent=2, ensure_ascii=False)

    # Save awards data separately for easier access
    awards_file = os.path.join(output_dir, f"sundt_awards_{timestamp}.json")
    with open(awards_file, 'w', encoding='utf-8') as f:
        json.dump(results['awards'], f, indent=2, ensure_ascii=False)

    # Save combined data for convenience
    combined_file = os.path.join(output_dir, f"sundt_combined_{timestamp}.json")
    with open(combined_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    logger.info("=== DATA SAVED ===")
    logger.info(f"Projects saved to: {projects_file}")
    logger.info(f"Awards saved to: {awards_file}")
    logger.info(f"Combined data saved to: {combined_file}")

    return {
        'projects_file': projects_file,
        'awards_file': awards_file,
        'combined_file': combined_file
    }

# Main execution block - only run when script is executed directly
if __name__ == "__main__":
    try:
        # Run the main crawler function
        results = crawl_site()

        logger.info("=== RESULTS ===")
        logger.info(f"Projects found: {len(results['projects'])}")
        logger.info(f"Awards found: {len(results['awards'])}")

        # Display sample results for verification
        if results['projects']:
            sample_project = results['projects'][0]
            logger.info(f"Sample project: {sample_project['project_name']}")
            logger.info(f"URL: {sample_project['project_url']}")
            if sample_project['metadata'].get('Location'):
                logger.info(f"Location: {sample_project['metadata']['Location']}")
            if sample_project['features']:
                logger.info(f"Features: {len(sample_project['features'])} items")
            if sample_project['similar_projects']:
                logger.info(f"Similar projects: {len(sample_project['similar_projects'])} found")
                for similar in sample_project['similar_projects'][:3]:  # Show first 3
                    logger.info(f"  - {similar['title']}: {similar['link']}")

        if results['awards']:
            sample_award = results['awards'][0]
            logger.info(f"Sample award: {sample_award['title']}")
            if sample_award['year']:
                logger.info(f"Year: {sample_award['year']}")
            if sample_award['awarded_by']:
                logger.info(f"Awarded by: {sample_award['awarded_by']}")
            if sample_award['project_name']:
                logger.info(f"Project: {sample_award['project_name']}")
            if sample_award['location']:
                logger.info(f"Location: {sample_award['location']}")


        # Save the crawled data if we have any results
        if results and (results['projects'] or results['awards']):
            saved_files = save_data_to_json(results)
            logger.info("Data ingestion phase complete!")
            logger.info(f"Total projects: {len(results['projects'])}")
            logger.info(f"Total awards: {len(results['awards'])}")
        else:
            logger.warning("No data to save - crawling may have failed")

    except Exception as e:
        logger.error(f"Error during crawling: {e}")
        logger.error("Make sure you have installed the required dependencies:")
        logger.error("pip install selenium chromedriver-autoinstaller")
        results = {'projects': [], 'awards': []}
