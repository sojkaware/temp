# Pretend you are the best computer scientist in the world and an expert on Scrapy. You will write a python program that will log in to an advanced SSO login webpage that uses SAML. Keep in mind that the login page is advanced, so the best way of doing this would be to use Puppeteer or Selenium, but this would be too resource heavy so you will do it only with Scrapy, you will avoid using Selenium or other similar resource heavy tools.  Specifically, it will do the following:
# 1. Open this webpage:
# url = 'https://moodle.fel.cvut.cz/auth/shibboleth/index.php'
# 2. The page will probably redirect so handle this. If you see a hidden tokens in the HTML after the redirects, for example SAML and others,  save them into a variable because you will need them later.
# 3. Fill out the username stored in a variable form_username into the form on the page into the field which has a HTML attribute name ="j_username". Fill out the password stored in a variable form_password into the form on the page into the field which has a HTML attribute name ="j_password". 
# 5. Submit the filled form. The confirm button of the SSO login form has a HTML attribute name="_eventId_proceed"
# 6. Check successful login by detecting the presence of the text "Welcome"




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



