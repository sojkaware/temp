import scrapy
import os
from urllib.parse import urlparse

class MySpider(scrapy.Spider):
    name = 'myspider'
    start_urls = ['https://example.com']

    def parse(self, response):
        # Check if login processing is enabled
        if self.settings.getbool('LOGIN_ENABLED', True):
            # Extract the CSRF token from the login form
            csrf_token = response.xpath('//input[@name="csrf_token"]/@value').extract_first()

            # Create a FormRequest with the login credentials
            yield scrapy.FormRequest(
                url='https://example.com/login',
                formdata={
                    'username': 'myusername',
                    'password': 'mypassword',
                    'csrf_token': csrf_token,
                },
                callback=self.after_login
            )
        else:
            # Login processing is disabled. Proceed to scrape the first page.
            self.logger.info("Login processing disabled. Scraping first page.")
            self.save_page(response)

            # Extract all the links on the page and scrape them
            self.logger.info("Scraping links on the first page.")
            for link in response.css('a::attr(href)').getall():
                yield response.follow(link, self.parse_link)

    def after_login(self, response):
        # Check if the login was successful
        if 'Welcome' in response.text:
            # Scrape the first page and save it to a file
            self.logger.info("Login successful. Scraping first page.")
            self.save_page(response)

            # Extract all the links on the page and scrape them
            self.logger.info("Scraping links on the first page.")
            for link in response.css('a::attr(href)').getall():
                yield response.follow(link, self.parse_link)

    def parse_link(self, response):
        # Scrape the linked page and save it to a file
        self.logger.info("Scraping linked page: %s", response.url)
        self.save_page(response)

        # Extract all the links on the linked page and scrape them
        self.logger.info("Scraping links on the linked page: %s", response.url)
        for link in response.css('a::attr(href)').getall():
            yield response.follow(link, self.parse_link)

    def save_page(self, response):
        # Extract the page content and save it to a file
        url_parts = urlparse(response.url)
        filename = url_parts.netloc + url_parts.path
        if filename.endswith('/'):
            filename = filename + 'index.html'
        else:
            filename = filename + '.html'
        save_path = os.path.join('output', filename)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'wb') as f:
            f.write(response.body)