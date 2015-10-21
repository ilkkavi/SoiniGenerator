__author__ = 'Ilkka'
from lxml import html
import requests


class SoiniParser(object):
    """ A tool to scrape the foreign minister's silly blog """
    posts = []

    def __init__(self, baseUrl="http://timosoini.fi/category/ploki/"):
        self.baseUrl = baseUrl

    def getBlogPosts(self, maxPages=0):
        self.__loadPages(maxPages)
        return self.posts

    def __loadPages(self, maxPages):
        response = requests.get(self.baseUrl)
        firstpage = html.fromstring(response.text)
        pagination = firstpage.xpath("//div[@class=\"pagination loop-pagination\"]")[0]
        pagecount = int(max(pagination.xpath("//a[@class=\"page-numbers\"]/text()")))
        pagecount = pagecount if maxPages == 0 else maxPages

        for pagenumber in range(1, pagecount+1):
            print "Scraping page %d... (%s)" % (pagenumber, (self.baseUrl + "/page/%d" % pagenumber))
            if pagenumber == 1:
                page = firstpage
            else:
                response = requests.get(self.baseUrl + "/page/%d" % pagenumber)
                page = html.fromstring(response.text)

            articles = page.xpath("//article")
            for article in articles:
                # Header
                header = article.xpath(".//h1[@class=\"entry-title\"]/a/text()")
                if not header:
                    header = article.xpath(".//h1[@class=\"entry-title\"]/a/span/text()")
                post = header[0].upper() + "\n\n"
                # Body
                post += "\n".join(article.xpath(".//div[@class=\"entry-content\"]/p/text()")).strip()
                self.posts.append(post)
