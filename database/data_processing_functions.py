import mwclient # library for working with the MediaWiki API for loading example Wikipedia articles
import mwparserfromhell # Parser for MediaWiki
import openai # will be used for tokenization
import re # for cutting links <ref> from Wikipedia articles
import tiktoken # to count tokens
from sections import SECTIONS_TO_IGNORE


CATEGORY_TITLE = "Category:Manchester United F.C."
WIKI_SITE = "en.wikipedia.org"
GPT_MODEL = "gpt-3.5-turbo" 

# Collect the titles of all articles
def titles_from_category(
    category: mwclient.listing.Category, # Set the typed parameter of the article category
    max_depth: int # Determine the depth of nesting articles
) -> set[str]:
    """Returns a set of page titles in a given Wikipedia category and its subcategories."""
    titles = set() # Use a set to store article titles
    for cm in category.members(): # Loop through nested category objects
        if type(cm) == mwclient.page.Page: # If the object is a page
            titles.add(cm.name) # add the page name to the titles storage
        elif isinstance(cm, mwclient.listing.Category) and max_depth > 0: # If the object is a category and the nesting depth has not reached the maximum
            deeper_titles = titles_from_category(cm, max_depth=max_depth - 1) # recursively call the function for the subcategory
            titles.update(deeper_titles) # add elements from another set to a set
    return titles

# The function returns a list of all subsections for a given section of a Wikipedia page

def all_subsections_from_section(
    section: mwparserfromhell.wikicode.Wikicode, # current section
    parent_titles: list[str], # Parent titles
    sections_to_ignore: set[str], # Sections to ignore
) -> list[tuple[list[str], str]]:
    """
    From a Wikipedia section returns a list of all nested sections.
    Each subsection is a tuple, where:
      - the first element is a list of parent sections, starting with the page title
      - the second element represents the section text
    """

    # Extract the headers of the current section
    headings = [str(h) for h in section.filter_headings()]
    title = headings[0]
    # Wikipedia titles are of the form: "== Heading =="

    if title.strip("=" + " ") in sections_to_ignore:
        # If the section header is in the ignore list, then skip it
        return []

    # Combine headings and subheadings to preserve context for chatGPT
    titles = parent_titles + [title]

    # Convert wikicode sections to string
    full_text = str(section)

    # Select the text of the section without a title
    section_text = full_text.split(title)[1]
    if len(headings) == 1:
        # If there is one header, then we form the resulting list
        return [(titles, section_text)]
    else:
        first_subtitle = headings[1]
        section_text = section_text.split(first_subtitle)[0]
        # Form the resulting list from the text up to the first subheading
        results = [(titles, section_text)]
        for subsection in section.get_sections(levels=[len(titles) + 1]):
            results.extend(
                # Call the function for getting nested sections for a given section
                all_subsections_from_section(subsection, titles, sections_to_ignore)
                ) # Combine the resulting lists of this function and the called one
        return results

# The function returns a list of all sections of the page, except those that are discarded
def all_subsections_from_title(
    title: str, # The title of a Wikipedia article that is parsed
    sections_to_ignore: set[str] = SECTIONS_TO_IGNORE, # Sections to ignore
    site_name: str = WIKI_SITE, # Link to the wikipedia site
) -> list[tuple[list[str], str]]:
    """
    From the title of the Wikipedia page returns a list of all nested sections.
    Each subsection is a tuple, where:
      - the first element is a list of parent sections, starting with the page title
      - the second element represents the section text
    """

    # Initialize the MediaWiki object
    # WIKI_SITE links to the English part of Wikipedia
    site = mwclient.Site(site_name)

    # We request the page by title
    page = site.pages[title]

    # Get the text representation of the page
    text = page.text()

    # Convenient parser for MediaWiki
    parsed_text = mwparserfromhell.parse(text)
    # Extract headers
    headings = [str(h) for h in parsed_text.filter_headings()]
    if headings: # If headings are found
        # As a summary, take the text up to the first heading
        summary_text = str(parsed_text).split(headings[0])[0]
    else:
        # If there are no headings, then the entire text is considered a summary
        summary_text = str(parsed_text)
    results = [([title], summary_text)] # Add a summary to the result list
    for subsection in parsed_text.get_sections(levels=[2]): # Retrieve the 2nd level sections
        results.extend(
            # Call the function for getting nested sections for a given section
            all_subsections_from_section(subsection, [title], sections_to_ignore)
        ) # Combine the resulting lists of this function and the called one
    return results

# Clean section text from <ref>xyz</ref> links, leading and trailing spaces
def clean_section(section: tuple[list[str], str]) -> tuple[list[str], str]:
    titles, text = section
    # Remove links
    text = re.sub(r"<ref.*?</ref>", "", text)
    # Remove spaces at the beginning and the end
    text = text.strip()
    return (titles, text)

# Filter out short and empty sections
def keep_section(section: tuple[list[str], str]) -> bool:
    """Returns True if the section should be kept, False otherwise."""
    titles, text = section
    # Filter by arbitrary length, you can choose another value
    if len(text) < 20:
        return False
    else:
        return True
    
# Token counting function
def num_tokens(text: str, model: str = GPT_MODEL) -> int:
    """Returns the number of tokens in a string."""
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))

# Line splitting function
def halved_by_delimiter(string: str, delimiter: str = "\n") -> list[str, str]:
    """Splits a string in two using a delimiter, attempting to balance the tokens on each side."""

    # Divide the line into parts using the delimiter, by default \n is a line break
    chunks = string.split(delimiter)
    if len(chunks) == 1:
        return [string, ""] # delimiter not found
    elif len(chunks) == 2:
        return chunks # no need to look for intermediate point
    else:
        # Counting tokens
        total_tokens = num_tokens(string)
        halfway = total_tokens // 2
        # Pre-split in the middle of the number of tokens
        best_diff = halfway
        # In the loop we look for which of the delimiters will be closest to best_diff
        for i, chunk in enumerate(chunks):
            left = delimiter.join(chunks[: i + 1])
            left_tokens = num_tokens(left)
            diff = abs(halfway - left_tokens)
            if diff >= best_diff:
                break
            else:
                best_diff = diff
        left = delimiter.join(chunks[:i])
        right = delimiter.join(chunks[i:])
        # Return the left and right parts of the optimally divided string
        return [left, right]


# The function truncates the string to the maximum number of tokens allowed
def truncated_string(
    string: str, # string
    model: str, # model
    max_tokens: int, # maximum number of tokens allowed
    print_warning: bool = True, # warning flag
) -> str:
    """Trim the string to the maximum number of tokens allowed."""
    encoding = tiktoken.encoding_for_model(model)
    encoded_string = encoding.encode(string)
    # Trim the string and decode it back
    truncated_string = encoding.decode(encoded_string[:max_tokens])
    if print_warning and len(encoded_string) > max_tokens:
        print(f"Warning!: String truncated from {len(encoded_string)} tokens to {max_tokens} tokens.")
    # Truncated string
    return truncated_string

# The function divides sections of the article into parts according to the maximum number of tokens
def split_strings_from_subsection(
    subsection: tuple[list[str], str], # sections
    max_tokens: int = 1000, # maximum number of tokens
    model: str = GPT_MODEL, # model
    max_recursion: int = 5, # maximum number of recursions
) -> list[str]:
    """
    Divides sections into a list of sections parts, each part contains no more than max_tokens.
    Each part is a tuple of parent headings [H1, H2, ...] and text (str).
    """
    titles, text = subsection
    string = "\n\n".join(titles + [text])
    num_tokens_in_string = num_tokens(string)
    # If the length is valid, it will return a string
    if num_tokens_in_string <= max_tokens:
        return [string]
    # if as a result of the recursion it was not possible to split the string, then we simply truncate it by the number of tokens
    elif max_recursion == 0:
        return [truncated_string(string, model=model, max_tokens=max_tokens)]
    # otherwise we will divide in half and perform recursion
    else:
        titles, text = subsection
        for delimiter in ["\n\n", "\n", "."]: # Trying to use delimiters from largest to smallest (break, paragraph, period)
            left, right = halved_by_delimiter(text, delimiter=delimiter)
            if left == "" or right == "":
                # if either half is empty, try again with a simpler separator
                continue
            else:
                # apply recursion on each half
                results = []
                for half in [left, right]:
                    half_subsection = (titles, half)
                    half_strings = split_strings_from_subsection(
                        half_subsection,
                        max_tokens=max_tokens,
                        model=model,
                        max_recursion=max_recursion - 1, # reduce the maximum number of recursions
                    )
                    results.extend(half_strings)
                return results
    # otherwise no split was found, so just truncate the line (should be very rare)
    return [truncated_string(string, model=model, max_tokens=max_tokens)]

