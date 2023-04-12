
# scrapy-splash allows you to run JavaScript code and render pages, which is useful for handling SSO login pages.

# First, install scrapy-splash:

# pip install scrapy-splash
# Then, follow these steps to create a Scrapy spider for the task:

# Create a new Scrapy project if you haven't already.
# scrapy startproject sso_login

# Start the splash Docker container:
# docker run -p 8050:8050 scrapinghub/splash
# Create a new spider file in the spiders directory and add the following code:
# Run the spider:
# scrapy crawl sso_login
