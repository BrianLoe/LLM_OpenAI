from bs4 import BeautifulSoup

class HTMLCleaner:
    def __init__(self, html_content):
        self.html_content = html_content
        self.cleaned_tags = None
        self.cleaned_html = None
        self.cleaned_content = None

    def remove_unwanted_tags(self, unwanted_tags=["script", "style"]):
        """
        This removes unwanted HTML tags from the given HTML content.
        """
        soup = BeautifulSoup(self.html_content, 'html.parser')

        for tag in unwanted_tags:
            for element in soup.find_all(tag):
                element.decompose()
                
        self.cleaned_tags = str(soup)
        return self.cleaned_tags

    def extract_tags(self, tags):
        """
        This takes in HTML content and a list of tags, and returns a string
        containing the text content of all elements with those tags, along with their href attribute if the
        tag is an "a" tag.
        """
        if not self.cleaned_tags:
            print("Please run remove unwanted tags first.")
            return None
        
        soup = BeautifulSoup(self.cleaned_tags, 'html.parser')
        text_parts = []

        for tag in tags:
            elements = soup.find_all(tag)
            for element in elements:
                # If the tag is a link (a tag), append its href as well
                if tag == "a":
                    href = element.get('href')
                    if href:
                        text_parts.append(f"{element.get_text()} ({href})")
                    else:
                        text_parts.append(element.get_text())
                else:
                    text_parts.append(element.get_text())
        cleaned_text_parts = [part for part in text_parts if len(part)>15]
        joined_parts = ' '.join(cleaned_text_parts)
        self.cleaned_html = joined_parts
        return self.cleaned_html

    def remove_uneccessary_lines(self):
        if not self.cleaned_html:
            print("Please run extract tags first.")
            return None
        # Split content into lines
        lines = self.cleaned_html.split("\n")

        # Strip whitespace for each line
        stripped_lines = [line.strip() for line in lines]

        # Filter out empty lines
        non_empty_lines = [line for line in stripped_lines if line]

        # Remove duplicated lines (while preserving order)
        seen = set()
        deduped_lines = [line for line in non_empty_lines if not (
            line in seen or seen.add(line))]

        # Join the cleaned lines without any separators (remove newlines)
        cleaned_content = "".join(deduped_lines)
        self.cleaned_content = cleaned_content
        return self.cleaned_content
    
    def clean_html_content(self, tags, unwanted_tags=["script", "style","a","iframe","footer"]):
        self.remove_unwanted_tags(unwanted_tags)
        self.extract_tags(tags)
        self.remove_uneccessary_lines()
        return self.cleaned_content